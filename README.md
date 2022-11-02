# noLogarithms

## A world without logarithms

$y = \log _b (x)$, answers the question to what power $y$ do I have to raise a basis $b$ to get $x$. 

Although the action of the logarithm is somewhat cumbersome, this does not prevent the function from appearing in many natural phenomena.  
Loudness, pitch, the strength of earthquakes are all measured in logarithmic scales. 

How would a world without logarithms look like? 

Here we listen to this world exploring the linear piano. 

### Setting up the environment and installation

This notebooks require python3, matplotlib, scipy, numpy, sounddevice, ipywidgets

Install them in a virtual environement with pip

```
pip install numpy matplotlib jupyterlab scipy ipywidgets sounddevice
```

You will also need [pyanito](https://github.com/floreencia/pyanito)

#### Install pyanito

1. Clone it from https://github.com/floreencia/pyanito
2. Create an virtual environment and install it there `pip install -e /path/to/package/pyanito`


<div class="alert alert-block alert-danger">
<b>Danger:</b> This notebook plays high pitch sounds. When running it, ensure the volume is low and  you’re not wearing headphones.
</div>


### TODO

Add note duration,  add songs, guitar and piano synthesisers. 
