(function () {

	var _ = require('./lib/lodash');
	var $m = require('./fastmath');

	var _ann = {};

	var MIN_NUM_LAYERS = 3;

	_ann.conf = {
		"rate": 0.1,
	};

	function Ann (sizes, activationFns) {
		if(activationFns && !_.isArray(activationFns)) {
			throw new Error("Init: needs an array of activation function tuples [fn, fnPrime]");
		} else if(!activationFns) {
			activationFns = [[sigmoid, sigmoidPrime]];
		}
 
		if(activationFns.length === 1) {
			sizes = _.map(sizes, function (s) {
				return [s, activationFns[0]];
			});
		} else if(activationFns.length !== (sizes.length - 1)) {
			throw new Error("Init: needs one tuple [fn, fnPrime] for each layer exept input layer (num layers - 1), or just one tuple for all layers");
		} else {
			sizes = [sizes[0]].concat(_.zip(_.rest(sizes), activationFns));
		}

		var inputsPerNeuron = sizes[0][0];

		//initialize layers
		this.layers = _.map(_.rest(sizes), function (sizeFnTuple) {
			var numNeurons = sizeFnTuple[0];
			var aFn = vectorize(sizeFnTuple[1][0]);
			var aFnPrime = vectorize(sizeFnTuple[1][1]);
			var layer = new Layer(numNeurons, inputsPerNeuron, aFn, aFnPrime);

			inputsPerNeuron = numNeurons; // num inputs for next layer

			return layer;
		});
	}

	function Layer (numNeurons, numInputsPerNeuron, activationFn, activationFnPrime) {
		// initialize weights 
		this.m_weights = $m.random(numNeurons, numInputsPerNeuron);
		this.m_biases = $m.random(numNeurons, 1);
		
		this.activations = [];
		this.partials = [];

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
	 * Initializes an Ann with randomized weights (normally distributed N~(0,1))
	 *
	 * @param {Array} sizes Array of integers, each value representing a layer and the number of nodes in that layer
	 * @returns {Ann} Returns a new Ann
	 */
	function init (sizes, activationFn) {
		if(sizes.length < MIN_NUM_LAYERS) {
			throw new Error("The network at least one input, one hidden and one output layer");
		}
		return new Ann(sizes, activationFn);
	}

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
					ann.layers[numLayers-1].partials);

			deltaW[numLayers-1] = $m.multMatrixMatrix(delta, $m.transpose(ann.layers[numLayers-2].activations));
			deltaB[numLayers-1] = delta;

			// backpropagate error and accumulate weight deltas
			for(var l = numLayers - 2; l >= 0; l--) {
				var _activation = l > 0 ? ann.layers[l - 1].activations : inputs;
				
				delta = $m.multMatrixElementwiseMutate($m.multMatrixMatrix($m.transpose(ann.layers[l+1].m_weights), delta),
					ann.layers[l].partials);

				deltaW[l] = $m.multMatrixMatrix(delta, $m.transpose(_activation));
				deltaB[l] = delta;
			}
				
			// update weights by gradient descent
			for(var j = 0; j < ann.layers.length; j++) {
				ann.layers[j].m_weights = $m.subtractMatrixMatrixMutate(ann.layers[j].m_weights,
					$m.multMatrixScalarMutate(deltaW[j], _ann.conf.rate));
				ann.layers[j].m_biases = $m.subtractMatrixMatrixMutate(ann.layers[j].m_biases,
					$m.multMatrixScalarMutate(deltaB[j], _ann.conf.rate));
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

	function predict (ann, inputs) {
		inputs = $m.transpose([inputs]);

		for (var i = 0; i < ann.layers.length; i++) {
			var z = $m.addMatrixMatrixMutate($m.multMatrixMatrix(ann.layers[i].m_weights, inputs), ann.layers[i].m_biases),
				activation = ann.layers[i].activationFn(z);

			ann.layers[i].partials = ann.layers[i].activationFnPrime(z);
			ann.layers[i].activations = activation;

			inputs = activation;
		}

		return inputs;
	}

	function costDerivative (activations, y) {
		return $m.subtractMatrixMatrix(activations, y);
	}

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

	function shuffle (data) {
		return _.shuffle(data);
	}

	_ann.init = init;
	_ann.train = train;
	_ann.predict = predict;
	_ann.sigmoid = sigmoid;
	_ann.sigmoidPrime = sigmoidPrime;
	_ann.gaussian = gaussian;
	_ann.gaussianPrime = gaussianPrime;
	_ann.tanh = tanh;
	_ann.tanhPrime = tanhPrime;
	_ann.shuffle = shuffle;

	module.exports = _ann;

}.call(this));

