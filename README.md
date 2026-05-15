# calochallenge_diffusion_model

Easy to train:
 put your dataset in `data/` folder, then run:
 `python train.py --dataset dataset_1 --particle photon ` or
 `python train.py --dataset dataset_2`

 other parameters can be helped by `python train.py --help`

 Easy to inference:
 after training, you can get your models in folder models, then run:
 ``` 
   python inference_dataset1.py \
       --checkpoint your_checkpoint_dir \
       --output output_dir \
       --xml data/binning_dataset_1_photons.xml \
       --particle your_particle \
       --num_samples  your_samples
       --distribution load --energy_file_dir your_reference_dir \
 ```
 or
 ```
    python inference.py \
       --checkpoint your_checkpoint_dir \
       --output output_dir \
       --xml data/binning_dataset_2/3.xml \
       --num_samples  your_samples
       --dataset dataset2/3
       --distribution load --energy_file_dir your_reference_dir \
 ```
