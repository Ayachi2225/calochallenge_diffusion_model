import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from schedule import VPSchedule
from net import UNet3D


class DDIMModel3D(nn.Module):

    def __init__(
        self, 
        t_dim: int = 128, 
        e_dim: int = 128,
        training_obj: str = 'noise_pred',  # 'noise_pred', 'mean_pred', 'hybrid'
        cold_diffusion: bool = False,
        E_bins: torch.Tensor = None,
        avg_showers: torch.Tensor = None,
        std_showers: torch.Tensor = None,
        cold_noise_scale: float = 1.0,
        use_mask: bool = False,  # 新增：是否使用mask
    ):
        super().__init__()
        self.schedule = VPSchedule()
        self.net = UNet3D(t_dim=t_dim, e_dim=e_dim)
        
        self.training_obj = training_obj
        supported = ['noise_pred', 'mean_pred', 'hybrid', 'score_pred']
        if self.training_obj not in supported:
            raise ValueError(f"training_obj 必须是 {supported} 之一，但得到 '{training_obj}'")
        
        self.cold_diffusion = cold_diffusion
        self.E_bins = E_bins
        self.avg_showers = avg_showers
        self.std_showers = std_showers
        self.cold_noise_scale = cold_noise_scale
        
        self.use_mask = use_mask  # 新增
        
        print(f"[Model] 训练目标: {self.training_obj}")
        print(f"[Model] 使用mask: {self.use_mask}")
        if self.cold_diffusion:
            print("[Model] 冷扩散模式已启用")
            if E_bins is None or avg_showers is None or std_showers is None:
                raise ValueError("冷扩散模式需要提供 E_bins, avg_showers, std_showers")

    def _broadcast(self, rate: torch.Tensor) -> torch.Tensor:
        return rate[:, None, None, None, None]

    def lookup_avg_std_shower(self, energies: torch.Tensor):
        """根据能量查找对应的平均和标准差 shower"""
        energies_flat = energies.squeeze()
        
        idxs = torch.bucketize(energies_flat, self.E_bins.to(energies.device)) - 1
        
        idxs = torch.clamp(idxs, 0, len(self.avg_showers) - 1)
        
        avg = self.avg_showers[idxs].to(energies.device)
        std = self.std_showers[idxs].to(energies.device)
        
        return avg, std

    def gen_cold_image(
        self, 
        energy: torch.Tensor, 
        cold_noise_scale: float = 1.0, 
        noise: torch.Tensor = None
    ) -> torch.Tensor:
        """生成冷扩散初始图像"""
        avg_shower, std_shower = self.lookup_avg_std_shower(energy)
        
        if noise is None:
            noise = torch.randn_like(avg_shower)
        
        # 公式: avg_shower + cold_noise_scale * (noise * std_shower)
        cold_image = avg_shower + cold_noise_scale * (noise * std_shower)
        
        return cold_image

    def get_loss(
        self, 
        x0: torch.Tensor, 
        energy: torch.Tensor,
        mask: torch.Tensor = None,  # 新增：mask参数
        energy_loss_scale: float = 0.0
    ) -> torch.Tensor:
        """
        计算损失
        Args:
            x0: 真实数据 [B, C, D, H, W]
            energy: 能量 [B, 1]
            mask: mask数组 [B, C, D, H, W]，1表示有效，0表示mask区域
            energy_loss_scale: 能量损失权重
        """
        B = x0.shape[0]
        device = x0.device
        
        t = torch.rand(B, device=device)
        
        sr, nr = self.schedule(t)  # signal_rate, noise_rate
        sr = self._broadcast(sr)
        nr = self._broadcast(nr)
        
        eps = torch.randn_like(x0)
        
        if self.cold_diffusion:
            avg_shower, std_shower = self.lookup_avg_std_shower(energy)
            eps = avg_shower + self.cold_noise_scale * (std_shower * torch.randn_like(x0))
        
        x_t = sr * x0 + nr * eps
        
        sigma2 = nr ** 2
        
        net_output = self.net(x_t, t, energy)
        
        # 计算基础loss
        if self.training_obj == 'noise_pred':
            target = eps
            pred = net_output
            loss_element = F.mse_loss(pred, target, reduction='none')
            
        elif self.training_obj == 'mean_pred':
            target = x0
            pred = net_output
            
            weight = 1.0 / (sigma2 + 1e-8)
            loss_element = weight * F.mse_loss(pred, target, reduction='none')
            
        elif self.training_obj == 'hybrid':
            c_skip = 1.0 / (sigma2 + 1.0)
            c_out = torch.sqrt(sigma2) / torch.sqrt(sigma2 + 1.0)

            x0_pred = c_skip * x_t + c_out * net_output

            target = x0
            pred = x0_pred

            weight = 1.0 + 1.0 / (sigma2 + 1e-8)
            loss_element = weight * F.mse_loss(pred, target, reduction='none')

        elif self.training_obj == 'score_pred':
            target = -eps
            pred   = nr * net_output
            loss_element = F.mse_loss(pred, target, reduction='none')
        
        # 应用mask（如果使用）
        if self.use_mask and mask is not None:
            # mask: [B, C, D, H, W]，1为有效区域，0为mask区域
            loss_element = loss_element * mask
            # 计算有效区域的平均loss
            num_valid = mask.sum() + 1e-8  # 避免除零
            loss = loss_element.sum() / num_valid
        else:
            # 不使用mask时，全局平均
            loss = loss_element.mean()
        
        # 能量损失（可选）
        if energy_loss_scale > 0:
            loss_energy = self._compute_energy_loss(x0, pred, energy_loss_scale, mask)
            loss = loss + loss_energy
        
        return loss
    
    def _compute_energy_loss(
        self, 
        x0_true: torch.Tensor, 
        x0_pred: torch.Tensor,
        scale: float,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算能量守恒损失
        Args:
            x0_true: 真实数据
            x0_pred: 预测数据
            scale: 损失权重
            mask: mask数组（可选）
        """
        dims = list(range(1, len(x0_true.shape)))
        
        if self.use_mask and mask is not None:
            energy_true = torch.sum(x0_true * mask, dim=dims)
            energy_pred = torch.sum(x0_pred * mask, dim=dims)
            nvoxels = torch.sum(mask, dim=dims) + 1e-8
        else:
            energy_true = torch.sum(x0_true, dim=dims)
            energy_pred = torch.sum(x0_pred, dim=dims)
            nvoxels = torch.prod(torch.tensor(x0_true.shape[1:], device=x0_true.device)).float()
        
        loss_energy = scale * F.mse_loss(energy_true, energy_pred) / nvoxels
        return loss_energy

    @torch.no_grad()
    def predict_x0_from_output(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:

        sr, nr = self.schedule(t)
        sr = self._broadcast(sr)
        nr = self._broadcast(nr)
        sigma2 = nr ** 2
        
        if self.training_obj == 'noise_pred':

            x0_pred = (x_t - nr * net_output) / sr
        
        elif self.training_obj == 'mean_pred':
            x0_pred = net_output
        
        elif self.training_obj == 'hybrid':
            c_skip = 1.0 / (sigma2 + 1.0)
            c_out = torch.sqrt(sigma2) / torch.sqrt(sigma2 + 1.0)
            x0_pred = c_skip * x_t + c_out * net_output

        elif self.training_obj == 'score_pred':
            # Tweedie公式: x0 = (x_t + σ² · s_θ) / sr
            x0_pred = (x_t + sigma2 * net_output) / sr

        return x0_pred

    @torch.no_grad()
    def predict_eps_from_output(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        sr, nr = self.schedule(t)
        sr = self._broadcast(sr)
        nr = self._broadcast(nr)
        sigma2 = nr ** 2
        
        if self.training_obj == 'noise_pred':
            eps_pred = net_output
        
        elif self.training_obj == 'mean_pred':
            x0_pred = net_output
            eps_pred = (x_t - sr * x0_pred) / (nr + 1e-8)
        
        elif self.training_obj == 'hybrid':
            c_skip = 1.0 / (sigma2 + 1.0)
            c_out = torch.sqrt(sigma2) / torch.sqrt(sigma2 + 1.0)
            x0_pred = c_skip * x_t + c_out * net_output
            eps_pred = (x_t - sr * x0_pred) / (nr + 1e-8)

        elif self.training_obj == 'score_pred':
            # score → noise: ε = -nr · s_θ
            eps_pred = -nr * net_output

        return eps_pred

    @torch.no_grad()
    def predict_score_from_output(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """从网络输出计算 score = ∇_x log p_t(x) = -ε/nr

        适用于所有训练目标，通过先计算 ε_pred 再转换。
        """
        eps_pred = self.predict_eps_from_output(x_t, net_output, t)
        _, nr = self.schedule(t)
        nr = self._broadcast(nr)
        return -eps_pred / (nr + 1e-8)

    @torch.no_grad()
    def transfer(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        net_output: torch.Tensor,
    ) -> torch.Tensor:

        t = t.expand(x.shape[0])
        t_next = t_next.expand(x.shape[0])

        x0_pred = self.predict_x0_from_output(x, net_output, t)
        eps_pred = self.predict_eps_from_output(x, net_output, t)
        
        sr_n, nr_n = self.schedule(t_next)
        sr_n = self._broadcast(sr_n)
        nr_n = self._broadcast(nr_n)

        x_next = sr_n * x0_pred + nr_n * eps_pred
        
        return x_next

    @torch.no_grad()
    def runge_kutta(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        ets: list,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        4阶Runge-Kutta方法
        """
        t_mid = (t + t_next) / 2
        B = x.shape[0]

        # k1
        e1 = self.net(x, t.expand(B), energy)
        ets.append(e1)

        # k2
        x2 = self.transfer(x, t, t_mid, e1)
        e2 = self.net(x2, t_mid.expand(B), energy)

        # k3
        x3 = self.transfer(x, t, t_mid, e2)
        e3 = self.net(x3, t_mid.expand(B), energy)

        # k4
        x4 = self.transfer(x, t, t_next, e3)
        e4 = self.net(x4, t_next.expand(B), energy)

        et = (1 / 6) * (e1 + 2 * e2 + 2 * e3 + e4)
        return et

    @torch.no_grad()
    def gen_order_4(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        ets: list,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        4阶PNDM采样
        """
        if len(ets) > 2:
            e_cur = self.net(x, t.expand(x.shape[0]), energy)
            ets.append(e_cur)
            
            net_output = (1 / 24) * (
                 55 * ets[-1]
               - 59 * ets[-2]
               + 37 * ets[-3]
               -  9 * ets[-4]
            )
        else:
            net_output = self.runge_kutta(x, t, t_next, ets, energy)

        x_next = self.transfer(x, t, t_next, net_output)
        if len(ets) > 4:
            ets[:] = ets[-4:]

        return x_next

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple,
        energy: torch.Tensor,
        num_steps: int = 50,
        device: str = 'cpu',
        cold_noise_scale: float = 1.0,
        eta: float = 0.0,  # DDIM参数，0=确定性，1=DDPM
    ) -> torch.Tensor:
        energy = energy.to(device)
        
        if self.cold_diffusion:
            avg_showers, std_showers = self.lookup_avg_std_shower(energy)
            noise = torch.randn(shape, device=device)
            x = avg_showers + cold_noise_scale * (std_showers * noise)
            
            print(f"[Cold Diffusion DDIM] 初始化范围: [{x.min():.4f}, {x.max():.4f}]")
        else:
            x = torch.randn(shape, device=device)
        
        ts = torch.linspace(0.999, 0.001, num_steps + 1, device=device)
        
        for step in tqdm(range(num_steps), desc="DDIM采样"):
            t = ts[step]
            t_next = ts[step + 1]
            
            net_output = self.net(x, t.expand(x.shape[0]), energy)
            
            x0_pred = self.predict_x0_from_output(x, net_output, t)
            
            eps_pred = self.predict_eps_from_output(x, net_output, t)
            
            sr_n, nr_n = self.schedule(t_next.expand(x.shape[0]))
            sr_n = self._broadcast(sr_n)
            nr_n = self._broadcast(nr_n)
            
            x_next = sr_n * x0_pred + nr_n * eps_pred
            
            if eta > 0 and step < num_steps - 1:
                sr_cur, nr_cur = self.schedule(t.expand(x.shape[0]))
                sr_cur = self._broadcast(sr_cur)
                nr_cur = self._broadcast(nr_cur)
                
                sigma = eta * torch.sqrt(
                    (nr_n ** 2) * (1 - sr_cur ** 2 / sr_n ** 2) / (1 - sr_cur ** 2)
                )
                
                noise = torch.randn_like(x)
                x_next = x_next + sigma * noise
            
            x = x_next
        
        print(f"[DDIM] 最终范围: [{x.min():.4f}, {x.max():.4f}]")
        return x

    @torch.no_grad()
    def pndm_sample(
        self,
        shape: tuple,
        energy: torch.Tensor,
        num_steps: int = 50,
        device: str = 'cpu',
        cold_noise_scale: float = 1.0,
    ) -> torch.Tensor:
        energy = energy.to(device)
        ets = []  # 存储历史网络输出
        
        if self.cold_diffusion:
            avg_showers, std_showers = self.lookup_avg_std_shower(energy)
            noise = torch.randn(shape, device=device)
            x = avg_showers + cold_noise_scale * (std_showers * noise)
            
            print(f"[Cold Diffusion PNDM] 初始化范围: [{x.min():.4f}, {x.max():.4f}]")
        else:
            x = torch.randn(shape, device=device)
        
        ts = torch.linspace(0.999, 0.001, num_steps + 1, device=device)
        
        for step in tqdm(range(num_steps), desc="PNDM采样"):
            t = ts[step]
            t_next = ts[step + 1]
            x = self.gen_order_4(x, t, t_next, ets, energy)
        
        print(f"[PNDM] 最终范围: [{x.min():.4f}, {x.max():.4f}]")
        return x

    # ========================================================================
    #  分数扩散采样方法 (Score-based SDE/ODE samplers)
    # ========================================================================

    @torch.no_grad()
    def euler_maruyama_sample(
        self,
        shape: tuple,
        energy: torch.Tensor,
        num_steps: int = 200,
        device: str = 'cpu',
        cold_noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Euler-Maruyama 离散化逆SDE采样

        逆SDE: dx = [-½β(t)x - β(t)∇_x log p_t(x)] dt + √β(t) dW̄
        离散化: x_{t-Δt} = x_t + ½β(t)x_t·Δt + β(t)·s_θ(x_t,t)·Δt + √(β(t)·Δt)·z
        """
        energy = energy.to(device)

        if self.cold_diffusion:
            avg_showers, std_showers = self.lookup_avg_std_shower(energy)
            noise = torch.randn(shape, device=device)
            x = avg_showers + cold_noise_scale * (std_showers * noise)
        else:
            x = torch.randn(shape, device=device)

        ts = torch.linspace(0.999, 0.001, num_steps + 1, device=device)

        for step in tqdm(range(num_steps), desc="Euler-Maruyama"):
            t_cur = ts[step]
            t_next = ts[step + 1]
            dt = t_cur - t_next  # Δt > 0

            beta_t = self.schedule.beta(t_cur.expand(x.shape[0]))
            beta_t = self._broadcast(beta_t)
            g_t = torch.sqrt(beta_t)

            net_output = self.net(x, t_cur.expand(x.shape[0]), energy)
            score = self.predict_score_from_output(x, net_output, t_cur.expand(x.shape[0]))

            # 确定性部分: drift
            drift = 0.5 * beta_t * x + beta_t * score

            # 随机部分: diffusion
            noise = torch.randn_like(x)
            diffusion = g_t * torch.sqrt(dt) * noise

            x = x + drift * dt + diffusion

        print(f"[Euler-Maruyama] 最终范围: [{x.min():.4f}, {x.max():.4f}]")
        return x

    @torch.no_grad()
    def probability_flow_ode_sample(
        self,
        shape: tuple,
        energy: torch.Tensor,
        num_steps: int = 200,
        device: str = 'cpu',
        cold_noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Probability Flow ODE 采样 (确定性)

        ODE: dx/dt = f(x,t) - ½g(t)²·∇_x log p_t(x)
        离散化: x_{t-Δt} = x_t + ½β(t)x_t·Δt + ½β(t)·s_θ(x_t,t)·Δt
        """
        energy = energy.to(device)

        if self.cold_diffusion:
            avg_showers, std_showers = self.lookup_avg_std_shower(energy)
            noise = torch.randn(shape, device=device)
            x = avg_showers + cold_noise_scale * (std_showers * noise)
        else:
            x = torch.randn(shape, device=device)

        ts = torch.linspace(0.999, 0.001, num_steps + 1, device=device)

        for step in tqdm(range(num_steps), desc="ProbFlow ODE"):
            t_cur = ts[step]
            t_next = ts[step + 1]
            dt = t_cur - t_next

            beta_t = self.schedule.beta(t_cur.expand(x.shape[0]))
            beta_t = self._broadcast(beta_t)

            net_output = self.net(x, t_cur.expand(x.shape[0]), energy)
            score = self.predict_score_from_output(x, net_output, t_cur.expand(x.shape[0]))

            drift = 0.5 * beta_t * x + 0.5 * beta_t * score
            x = x + drift * dt

        print(f"[ProbFlow ODE] 最终范围: [{x.min():.4f}, {x.max():.4f}]")
        return x

    @torch.no_grad()
    def pc_sample(
        self,
        shape: tuple,
        energy: torch.Tensor,
        num_steps: int = 200,
        device: str = 'cpu',
        cold_noise_scale: float = 1.0,
        n_correct: int = 1,
        delta: float = 0.17,
    ) -> torch.Tensor:
        """Predictor-Corrector 采样

        Predictor: Euler-Maruyama SDE step
        Corrector: Langevin MCMC (退火朗之万动力学)

        Args:
            n_correct: 每个预测步后的朗之万修正次数
            delta: 朗之万步长系数 (ε_l = δ · nr(t)²)
        """
        energy = energy.to(device)

        if self.cold_diffusion:
            avg_showers, std_showers = self.lookup_avg_std_shower(energy)
            noise = torch.randn(shape, device=device)
            x = avg_showers + cold_noise_scale * (std_showers * noise)
        else:
            x = torch.randn(shape, device=device)

        ts = torch.linspace(0.999, 0.001, num_steps + 1, device=device)

        for step in tqdm(range(num_steps), desc="Predictor-Corrector"):
            t_cur = ts[step]
            t_next = ts[step + 1]
            dt = t_cur - t_next

            # ---- Predictor: Euler-Maruyama ----
            beta_t = self.schedule.beta(t_cur.expand(x.shape[0]))
            beta_t = self._broadcast(beta_t)
            g_t = torch.sqrt(beta_t)

            net_output = self.net(x, t_cur.expand(x.shape[0]), energy)
            score = self.predict_score_from_output(x, net_output, t_cur.expand(x.shape[0]))

            drift = 0.5 * beta_t * x + beta_t * score
            pred_noise = torch.randn_like(x)
            x = x + drift * dt + g_t * torch.sqrt(dt) * pred_noise

            # ---- Corrector: Langevin MCMC ----
            for _ in range(n_correct):
                _, nr_next = self.schedule(t_next.expand(x.shape[0]))
                nr_next = self._broadcast(nr_next)

                net_output = self.net(x, t_next.expand(x.shape[0]), energy)
                score = self.predict_score_from_output(x, net_output, t_next.expand(x.shape[0]))

                epsilon_l = delta * (nr_next ** 2)
                langevin_noise = torch.randn_like(x)

                x = x + epsilon_l * score + torch.sqrt(2 * epsilon_l) * langevin_noise

        print(f"[Predictor-Corrector] 最终范围: [{x.min():.4f}, {x.max():.4f}]")
        return x

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        energy: torch.Tensor,
        num_steps: int = 50,
        device: str = 'cpu',
        cold_noise_scale: float = 1.0,
        method: str = 'pndm',
        eta: float = 0.0,  # DDIM only
        n_correct: int = 1,  # PC only
        delta: float = 0.17,  # PC only
    ) -> torch.Tensor:

        if method == 'pndm':
            return self.pndm_sample(shape, energy, num_steps, device, cold_noise_scale)
        elif method == 'ddim':
            return self.ddim_sample(shape, energy, num_steps, device, cold_noise_scale, eta)
        elif method == 'euler_maruyama':
            return self.euler_maruyama_sample(shape, energy, num_steps, device, cold_noise_scale)
        elif method == 'prob_flow':
            return self.probability_flow_ode_sample(shape, energy, num_steps, device, cold_noise_scale)
        elif method == 'pc':
            return self.pc_sample(shape, energy, num_steps, device, cold_noise_scale, n_correct, delta)
        else:
            raise ValueError(
                f"不支持的采样方法: {method}，"
                f"请选择 'pndm', 'ddim', 'euler_maruyama', 'prob_flow', 'pc'"
            )