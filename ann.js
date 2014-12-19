(function () {

    var $m = require('./math');

    var MIN_NUM_LAYERS = 3;
    var ERRORS = {
        LAYER_AFN_MISMATCH: 'Needs one tuple [fn, fnPrime] for each layer except the input layer, or just one tuple for all layers',
        TOO_FEW_LAYERS: 'The network at least one input, one hidden and one output layer'
    };

    var feedForwardAnn = {
        conf: {
            'rate': 0.1
        }
    };

    function Ann (sizes, activationFns) {
        var sizeFnTuples = [];
        var aFns = activationFns || [[sigmoid, sigmoidPrime]];

        if(aFns.length > 1 && aFns.length !== (sizes.length - 1)) {
            throw new Error(ERRORS.LAYER_AFN_MISMATCH);
        }

        this.layers = [];

        for(var i = 1; i < sizes.length; i++) {
            var numNeurons = sizes[i];
            var inputsPerNeuron = sizes[i-1];
            var aFn = vectorize(aFns.length > 1 ? aFns[i-1][0] : aFns[0][0]);
            var aFnPrime = vectorize(aFns.length > 1 ? aFns[i-1][1] : aFns[0][1]);

            this.layers[i-1] = new Layer(numNeurons, inputsPerNeuron, aFn, aFnPrime);
        }
    }

    function Layer (numNeurons, numInputsPerNeuron, activationFn, activationFnPrime) {
        this.m_weights = $m.random(numNeurons, numInputsPerNeuron);
        this.m_biases = $m.random(numNeurons, 1);
        
        this.activations = [];
        this.aPrimes = [];

        this.activationFn = activationFn;
        this.activationFnPrime = activationFnPrime;
    }

    /**
     * Vectorize a function so that when it can be called with a vector of values,
     * applying itself to each value
     *
     * @param {Function} fn Function to vectorize
     * @returns {Function} Returns vectorized function
     */
    function vectorize (fn) {
        return function (vec) {
            var len = vec.length;
            var res = [];
            for(var r = 0; r < len; r++) {
                res[r] = [fn(vec[r][0])];
            }
            return res;
        };
    }

    /**
     * Initializes an Ann with randomized weights (normally distributed from N~(0,1))
     *
     * @param {Array}   sizes Array of integers, each value representing a layer and the number of nodes in that layer
     * @param {Array}   activationFn Array of Function tuples, one tuple for each layer in 'sizes'. Each tuple contains 
     *                  an activation funtion and the derivative of that function.
     * @returns {Ann}   Returns a new Ann
     */
    function init (sizes, activationFn) {
        if(sizes.length < MIN_NUM_LAYERS) {
            throw new Error(ERRORS.TOO_FEW_LAYERS);
        }
        return new Ann(sizes, activationFn);
    }

    /**
     * Trains an Ann with specified data by calculating output error, performing back propagation of error
     * and adjusting weights using stochastic gradient descent.
     *
     * @param {Ann} ann Ann to train
     * @param {Array} trainingData array of objects with x and y properties, representing a training example 'x'
     * with output 'y' for that example.
     * @returns {number} returns 1 if training succeeds
     */
    function train (ann, trainingData) {

        for(var b = 0, trainlen = trainingData.length; b < trainlen; b++) {
            var x = trainingData[b].x,
                y = trainingData[b].y,
                numLayers = ann.layers.length,
                deltaW = [],
                deltaB = [],
                inputs = $m.transpose([x]),
                outputs = $m.transpose([y]);

            // feedforward pass
            var finalActivation = predict(ann, x);

            // calculate output error -> (a - y) * theta'(a(L))
            var delta = $m.multMatrixElementwiseMutate(costDerivative(finalActivation, outputs),
                    ann.layers[numLayers-1].aPrimes);

            deltaW[numLayers-1] = $m.multMatrixMatrix(delta, $m.transpose(ann.layers[numLayers-2].activations));
            deltaB[numLayers-1] = delta;

            // backpropagate error and accumulate weight deltas
            for(var l = numLayers - 2; l >= 0; l--) {
                var _activation = l > 0 ? ann.layers[l - 1].activations : inputs;
                
                delta = $m.multMatrixElementwiseMutate($m.multMatrixMatrix($m.transpose(ann.layers[l+1].m_weights), delta),
                    ann.layers[l].aPrimes);

                deltaW[l] = $m.multMatrixMatrix(delta, $m.transpose(_activation));
                deltaB[l] = delta;
            }
                
            // update weights by gradient descent
            for(var j = 0; j < ann.layers.length; j++) {
                ann.layers[j].m_weights = $m.subtractMatrixMatrixMutate(ann.layers[j].m_weights,
                    $m.multMatrixScalarMutate(deltaW[j], feedForwardAnn.conf.rate));
                ann.layers[j].m_biases = $m.subtractMatrixMatrixMutate(ann.layers[j].m_biases,
                    $m.multMatrixScalarMutate(deltaB[j], feedForwardAnn.conf.rate));
            }
        }

        // TODO: square error of predictions
        /*var squareError = [];
        for(var i = 0; i < trainingData.length; i++) {
            squareError[i] = costDerivative(ann.layers[ann.layers.length-1].activations, trainingData[i].y);
            for(var j = 0; j < squareError[i].length; j++) {
                squareError[i][j][0] = Math.pow(squareError[i][j][0], 2);
            }
        }*/

        return 1;
    }

    /**
     * Performs feedforward of an Ann. Takes input and calculates the output of the network.
     * Also updates the 
     *
     * @param {Ann} ann Ann to calcute output from
     * @param {Array} vector of input values
     * @returns {Array} Returns the network output
     */
    function predict (ann, inputs) {
        inputs = $m.transpose([inputs]);

        for (var i = 0; i < ann.layers.length; i++) {
            // calculate weighed input of neurons to this layer
            var z = $m.addMatrixMatrixMutate($m.multMatrixMatrix(ann.layers[i].m_weights, inputs), ann.layers[i].m_biases);

            var activation = ann.layers[i].activationFn(z);
            
            ann.layers[i].aPrimes = ann.layers[i].activationFnPrime(z); // measures how fast the activation function is changing in this layer
            ann.layers[i].activations = activation;

            inputs = activation;
        }

        return inputs;
    }

    /**
     * Vectorized quadratic cost function derivative
     */
    function costDerivative (activations, y) {
        return $m.subtractMatrixMatrix(activations, y);
    }

    /**
     * Predefined activation functions and their derivatives
     */
    function sigmoid (z) {
        p = 1.0;
        return 1/(1+Math.exp((-z)/p));
    }

    function sigmoidPrime (z) {
        return sigmoid(z)*(1-sigmoid(z));
    }

    function tanh (z) {
        return (Math.exp(z) - Math.exp(-z)) / (Math.exp(z) + Math.exp(-z));
    }

    function tanhPrime (z) {
        return 1-(tanh(z)*tanh(z));
    }

    function gaussian (z) {
        return Math.exp((-1)*z * z);
    }

    function gaussianPrime (z) {
        return (-2)*z*gaussian(z);
    }

    //+ Jonas Raoni Soares Silva
    //@ http://jsfromhell.com/array/shuffle [v1.0]
    function shuffle(o){ //v1.0
        for(var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
        return o;
    }

    feedForwardAnn.init = init;
    feedForwardAnn.train = train;
    feedForwardAnn.predict = predict;
    feedForwardAnn.sigmoid = sigmoid;
    feedForwardAnn.sigmoidPrime = sigmoidPrime;
    feedForwardAnn.gaussian = gaussian;
    feedForwardAnn.gaussianPrime = gaussianPrime;
    feedForwardAnn.tanh = tanh;
    feedForwardAnn.tanhPrime = tanhPrime;
    feedForwardAnn.shuffle = shuffle;

    module.exports = feedForwardAnn;

}.call(this));

