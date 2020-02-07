# WorldModels
Implementation of World Models paper


## Data generation
Initially we collect rollouts of the environment using a random policy.

```
python datasets/carracing.py
```
Generated rollouts will be placed in ```random``` directory in ```data/carracing``` folder

## Training

Later we train all the three modules i.e., VAE module, Memory module and Controller module independently

### Training the VAE module

```
python src/train_convVAE.py
```

### Training the Memory module

```
python src/train_lstm.py
```

### Training the Controller module

```
python src/evolution_pooling.py
```

## Testing

To test average rewards in gym environment
```
python src/test_gym.py
```

## Visualisation

To visualise vae original and reconstructed images for analysis
```
python utils/visualize_vae.py
```

To plot the graphs from the controller training for analysis
```
Run utils/plot_utility.ipynb file
```

## Logging

Logging from model training are available in ```logs``` folder

### Saved Checkpoints

Saved model checkpoints are available in ```checkpoints``` folder
