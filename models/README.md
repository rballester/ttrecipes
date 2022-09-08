# README

This folder contains the four models used in the paper *"High-dimensional Scalar Function Visualization Using Principal Parameterizations"* (R. Ballester-Ripoll, G. Halter, R. Pajarola):

- **Nassau County** (size 2^6)
- **Robot Arm** (size 64^8)
- **Damped Oscillator** (size 64^8)
- **Ebola Spread** (size 64^8)

See the paper for description and references of these models.

### Loading a Model

To load a model, run in Python:

```
import ttrecipes as tr
t = tr.core.load('name_of_the_model.npz')
```

### Code Used to Generate the Models

The code can be found in the file *models.py* from this repository.