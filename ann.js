(function () {

	var math = require('./lib/math');
	var _ = require('./lib/lodash');
	var sink = {};
	var $m = require('./fastmath')

	var SINK_LOGLEVEL_NONE = 0;
	var SINK_LOGLEVEL_FULL = 1;

	sink.conf = {
		"rate": 0.5,
		"momentum": 0.9,
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
		return math.subtract(activations, y);
	}

	function backProp (ann, x, y) {

		// feedforward pass
		var lastActivation = $m.transpose([x]);
		var activations = [lastActivation];
		var zs = [];
		var numLayers = ann.layers.length;

		ann.layers.forEach(function (layer) {
			var z = $m.addMatrixMatrixMutate($m.multMatrixMatrix(layer.m_weights, lastActivation), layer.m_biases);
			//if(z._size.length < 2) { z = math.matrix([z]); }
			zs.push(z);

			lastActivation = layer.activationFn(z);
			activations.push(lastActivation);
		});

		// calculate output error
		// (a - y) * theta'(a(L))
		var delta = $m.multMatrixElementwiseMutate(costDerivative(activations[activations.length-1], y), ann.layers[ann.layers.length-1].activationFnPrime(zs[zs.length-1]));
		//var err = costDerivative(_.last(activations),y);

		var delta_b_output = delta;
		var delta_w_output = $m.multMatrixMatrix(delta, $m.transpose(activations[activations.length-2]));

		ann.layers[numLayers-1].m_delta_b = $m.addMatrixMatrixMutate(ann.layers[numLayers-1].m_delta_b, delta_b_output);
		ann.layers[numLayers-1].m_delta_w = $m.addMatrixMatrixMutate(ann.layers[numLayers-1].m_delta_w, delta_w_output);

		// backpropagate error and accumulate weight deltas
		for(var l = numLayers - 2; l >= 0; l--) {
			var oldDelta = delta;
			delta = $m.multMatrixElementwiseMutate($m.multMatrixMatrix($m.transpose(ann.layers[l+1].m_weights), oldDelta),
				ann.layers[l].activationFnPrime(zs[l]));

			var delta_w = $m.multMatrixMatrix(delta, $m.transpose(activations[l - 1 + 1]));
			var delta_b = delta;

			ann.layers[l].m_delta_w = $m.addMatrixMatrixMutate(ann.layers[l].m_delta_w, delta_w);
			ann.layers[l].m_delta_b = $m.addMatrixMatrixMutate(ann.layers[l].m_delta_b, delta_b);
		}
		
		return 0; //math.squeeze(err);
	}

	sink.Layer = function (numNeurons, numInputsPerNeuron, activationFn, activationFnPrime) {
		this.m_weights = $m.random(numNeurons, numInputsPerNeuron);
		this.m_biases = $m.random(numNeurons, 1);

		this.activationFn = activationFn;
		this.activationFnPrime = activationFnPrime;
	};

	sink.Ann = function (sizes, activationFns) {

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

		var inputsPerNeuron = _.first(_.first(sizes));

		this.avgErr = 1;
		this.layers = _.map(_.rest(sizes), function (sizeFnTuple) {
			var numNeurons = sizeFnTuple[0];
			var aFn = vectorize(sizeFnTuple[1][0]);
			var aFnPrime = vectorize(sizeFnTuple[1][1]);
			var layer = new sink.Layer(numNeurons, inputsPerNeuron, aFn, aFnPrime);
			inputsPerNeuron = numNeurons; // num inputs for next layer
			return layer;
		});
	};

	sink.init = function (sizes, activationFn) {
		if(sizes < 3) {
			console.error("The network at least one input, one hidden and one output layer");
			return undefined;
		}
		return new sink.Ann(sizes, activationFn);
	};

	sink.sigmoid = function (z) {
		p = 1.0;
		return 1/(1+math.exp((-z)/p));
	};

	sink.sigmoidPrime = function (z) {
		return sink.sigmoid(z)*(1-sink.sigmoid(z));
	};

	/*sink.tanh = function (z) {
		return 2/(1+math.exp((-2) * z)) - 1;
	};*/

	sink.tanh = function (z) {
		return (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z));
	};

	sink.tanhPrime = function (z) {
		return 1-(sink.tanh(z)*sink.tanh(z));
	};

	sink.gaussian = function (z) {
		return math.exp((-1)*z * z);
	};

	sink.gaussianPrime = function (z) {
		return (-2)*z*sink.gaussian(z);
	};

	sink.train = function (ann, trainingData, numEpochs) {
		var averageError;

		for(var ep = 0; ep < numEpochs; ep++) {
			var now = Date.now();

			trainingData = _.shuffle(trainingData);

			var trainlen = trainingData.length;
			for(var b = 0; b < trainlen; b++) {

				averageError = 0;

				// initialize delta weight variables
				for(var i = 0; i < ann.layers.length; i++) {
					ann.layers[i].m_delta_w = $m.zeros(ann.layers[i].m_weights.length,
						ann.layers[i].m_weights[0].length);
					ann.layers[i].m_delta_b = $m.zeros(ann.layers[i].m_biases.length, 1);
				}
				
				var err = backProp(ann, trainingData[b].x, trainingData[b].y);

				for(var j = 0; j < ann.layers.length; j++) {
					var m_delta_rw = $m.multMatrixScalarMutate(ann.layers[j].m_delta_w, sink.conf.rate);
					var m_delta_rb = $m.multMatrixScalarMutate(ann.layers[j].m_delta_b, sink.conf.rate);

					ann.layers[j].m_weights = $m.subtractMatrixMatrixMutate(ann.layers[j].m_weights, m_delta_rw);
					ann.layers[j].m_biases = $m.subtractMatrixMatrixMutate(ann.layers[j].m_biases, m_delta_rb);
				}
			}

			var then = Date.now();

			if(sink.conf.logging === SINK_LOGLEVEL_FULL) {
				//console.log('\033[2J');
				console.log('Epoch ' + (ep+1) + ' completed');
				console.log("time spent: " + (then - now));
				//console.log('Average error: ' + ann.avgErr);
			}
		}

		return 1;
	};

	sink.classify = function (ann, inputs) {
		inputs = $m.transpose([inputs]);
		var inp = inputs;

		for (var i = 0; i < ann.layers.length; i++) {
			var oldInputs = inputs;
			inputs = ann.layers[i].activationFn($m.addMatrixMatrixMutate($m.multMatrixMatrix(ann.layers[i].m_weights, oldInputs),
				ann.layers[i].m_biases));
		}

		var outp = inputs;

		if(sink.conf.logging === SINK_LOGLEVEL_FULL) {
			//console.log("Input -> " + math.format(inp, 14));
			//console.log("Output -> " + math.format(outp, 14) + "\n");
		}

		return outp;
	};

	sink.testSuite = {};
	sink.testSuite.vectorize = vectorize;
	sink.testSuite.costDerivative = costDerivative;
	sink.testSuite.backProp = backProp;

	module.exports = sink;

}.call(this));

