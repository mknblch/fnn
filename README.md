# Feedforward neural network P.o.C.

Simple implementation of a [FNN](https://en.wikipedia.org/wiki/Feedforward_neural_network) 
with back-propagation to recap some AI knowledge. 

## Usage

```
    // crate a data set with input- and output-values
    DataSet train = DataSet.fromArray(                      // syntactic sugar
        new double[][] { new double[]{ ... }, ... },        // inputs
        new double[][] { new double[]{ ... }, ... }         // expected
    );

    FNN net = Trainer.builder( inputUnits , outputUnits )   // create a Builder
            .withLearningRate( learningRate )               // do setup
            .addHiddenLayer( hiddenUnits )                  // add hidden layers
            .build()                                        // build a Trainer
            .train( train, error, maxIterations );          // do training 

    double[] results = net.eval( new double[] { ... } )     // evaluate against input
```

Additional examples can be found in the unit tests.

## ToDo

- `Trainer.error(..)` might be buggy.