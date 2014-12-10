(function () {

	var _ = require('./lib/lodash');
	var sink = {};
	var $m = require('./fastmath')

	var SINK_LOGLEVEL_NONE = 0;
	var SINK_LOGLEVEL_FULL = 1;

	sink.conf = {
		"rate": 0.1,
		"logging": SINK_LOGLEVEL_FULL
	};

	function vectorize (fn) {
		return function (vec) {
			var len = vec.length;
			var res = [];
			for(var r = 0; r < len; r++) {
				res[r] = [fn(vec[r][0])];
			}
			return res;
		}
	}

	function costDerivative (activations, y) {
		return $m.subtractMatrixMatrix(activations, y);
	}

	sink.sigmoid = function (z) {
		p = 1.0;
		return 1/(1+Math.exp((-z)/p));
	};

	sink.sigmoidPrime = function (z) {
		return sink.sigmoid(z)*(1-sink.sigmoid(z));
	};

	sink.tanh = function (z) {
		return (Math.exp(z) - Math.exp(-z)) / (Math.exp(z) + Math.exp(-z));
	};

	sink.tanhPrime = function (z) {
		return 1-(sink.tanh(z)*sink.tanh(z));
	};

	sink.gaussian = function (z) {
		return Math.exp((-1)*z * z);
	};

	sink.gaussianPrime = function (z) {
		return (-2)*z*sink.gaussian(z);
	};

	sink.Ann = function (sizes, activationFns) {
		this.avgErr = 1;

		if(activationFns && !_.isArray(activationFns)) {
			console.error("Init: needs an array of activation function tuples [fn, fnPrime]");
			return undefined;
		} else if(!activationFns) {
			activationFns = [[sink.sigmoid, sink.sigmoidPrime]];
		}
 
		if(activationFns.length === 1) {
			sizes = _.map(sizes, function (s) {
				return [s, activationFns[0]];
			});
		} else if(activationFns.length !== (sizes.length - 1)) {
			console.error("Init: needs one tuple [fn, fnPrime] for each layer exept input layer (num layers - 1), or just one tuple for all layers");
			return undefined;
		} else {
			sizes = [sizes[0]].concat(_.zip(_.rest(sizes), activationFns));
		}

		var inputsPerNeuron = sizes[0][0];

		//initialize layers
		this.layers = _.map(_.rest(sizes), function (sizeFnTuple) {
			var numNeurons = sizeFnTuple[0];
			var aFn = vectorize(sizeFnTuple[1][0]);
			var aFnPrime = vectorize(sizeFnTuple[1][1]);
			var layer = new sink.Layer(numNeurons, inputsPerNeuron, aFn, aFnPrime);

			inputsPerNeuron = numNeurons; // num inputs for next layer

			return layer;
		});
	};

	sink.Layer = function (numNeurons, numInputsPerNeuron, activationFn, activationFnPrime) {
		// initialize weights 
		this.m_weights = $m.random(numNeurons, numInputsPerNeuron);
		this.m_biases = $m.random(numNeurons, 1);
		
		this.activations = [];
		this.partials = [];

		this.activationFn = activationFn;
		this.activationFnPrime = activationFnPrime;
	};

	sink.init = function (sizes, activationFn) {
		if(sizes.length < 3) {
			console.error("The network at least one input, one hidden and one output layer");
			return undefined;
		}
		return new sink.Ann(sizes, activationFn);
	};	

	sink.train = function (ann, trainingData) {
		var now = Date.now();

		trainingData = _.shuffle(trainingData);
		
		for(var b = 0, trainlen = trainingData.length; b < trainlen; b++) {
			var x = trainingData[b].x,
				y = trainingData[b].y,
				numLayers = ann.layers.length,
				deltaW = [],
				deltaB = [],
				inputs = $m.transpose([x]),
				outputs = $m.transpose([y]);

			// feedforward pass
			var finalActivation = sink.predict(ann, x);

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
					$m.multMatrixScalarMutate(deltaW[j], sink.conf.rate));
				ann.layers[j].m_biases = $m.subtractMatrixMatrixMutate(ann.layers[j].m_biases, 
					$m.multMatrixScalarMutate(deltaB[j], sink.conf.rate));
			}
		}

		// square error of predictions
		/*var squareError = [];
		for(var i = 0; i < trainingData.length; i++) {
			squareError[i] = costDerivative(ann.layers[ann.layers.length-1].activations, trainingData[i].y);
			for(var j = 0; j < squareError[i].length; j++) {
				squareError[i][j][0] = Math.pow(squareError[i][j][0], 2);
			}
		}*/

		var then = Date.now();

		if(sink.conf.logging === SINK_LOGLEVEL_FULL) {
			console.log('Epoch completed');
			console.log("time spent: " + (then - now));
			//console.log('Average error: ' + ann.avgErr);
		}

		return 1;
	};

	sink.predict = function (ann, inputs) {
		inputs = $m.transpose([inputs]);

		for (var i = 0; i < ann.layers.length; i++) {
			var z = $m.addMatrixMatrixMutate($m.multMatrixMatrix(ann.layers[i].m_weights, inputs), ann.layers[i].m_biases),
				activation = ann.layers[i].activationFn(z);

			ann.layers[i].partials = ann.layers[i].activationFnPrime(z);
			ann.layers[i].activations = activation;

			inputs = activation;
		}

		return inputs;
	};

	sink.testSuite = {};
	sink.testSuite.vectorize = vectorize;
	sink.testSuite.costDerivative = costDerivative;

	module.exports = sink;

}.call(this));

