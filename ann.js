(function () {

	var math = require('./lib/math');
	var _ = require('./lib/lodash');
	var sink = {};

	var SINK_LOGLEVEL_NONE = 0;
	var SINK_LOGLEVEL_FULL = 1;

	sink.conf = {
		"rate": 0.5,
		"momentum": 0.9,
		"logging": SINK_LOGLEVEL_FULL
	};

	function vectorize (fn) {
		return function (vec) {
			if(!vec.map) {
				return fn(vec);
			}
			return vec.map(function (val, idx, matrix) {
				return fn(val);
			});
		};
	}

	function hadamard (v1, v2) {
		if(!v2.valueOf()) {
			return v1[0] * v2;
		}
		if(!_.isEqual(v1.size(),v2.size())) {
			console.error("cannot multiply two vectors of different length");
			return math.matrix([]);
		}
		for (i = 0, len = v1.valueOf().length; i < len; i++) {
			v1.valueOf()[i][0] = v1.valueOf()[i][0] * v2.valueOf()[i][0];
		}
		return v1;
	}

	function costDerivative (activations, y) {
		return math.subtract(activations, y);
	}

	function backProp (ann, x, y) {

		// feedforward pass
		var lastActivation = math.transpose(math.matrix([x]));
		var activations = [lastActivation];
		var zs = [];
		var numLayers = ann.layers.length;

		ann.layers.forEach(function (layer) {
			var z = math.add(math.multiply(layer.m_weights, lastActivation), layer.m_biases);
			if(z._size.length < 2) { z = math.matrix([z]); }
			zs.push(z);
			lastActivation = layer.activationFn(z);
			activations.push(lastActivation);
		});

		// calculate output error
		// (a - y) * theta'(a(L))

		var delta = hadamard(costDerivative(_.last(activations), y), _.last(ann.layers).activationFnPrime(_.last(zs)));
		var err = costDerivative(_.last(activations),y);

		var delta_b_output = delta;
		var delta_w_output = math.multiply(delta, math.transpose(_.last(_.initial(activations))));

		ann.layers[numLayers-1].m_delta_b = math.add(ann.layers[numLayers-1].m_delta_b, delta_b_output);
		ann.layers[numLayers-1].m_delta_w = math.add(ann.layers[numLayers-1].m_delta_w, delta_w_output);

		// backpropagate error
		for(var l = numLayers - 2; l >= 0; l--) {
			var oldDelta = delta;
			delta = hadamard(math.multiply(math.transpose(ann.layers[l+1].m_weights), oldDelta),
				ann.layers[l].activationFnPrime(zs[l]));

			var delta_w = math.multiply(delta, math.transpose(activations[l - 1 + 1]));
			var delta_b = delta;

			ann.layers[l].m_delta_w = math.add(ann.layers[l].m_delta_w, delta_w);
			ann.layers[l].m_delta_b = math.add(ann.layers[l].m_delta_b, delta_b);
		}
		
		return math.squeeze(err);
	}

	sink.Layer = function (numNeurons, numInputsPerNeuron, activationFn, activationFnPrime) {
		this.m_weights = math.matrix(math.random([numNeurons, numInputsPerNeuron], -1, 1));
		this.m_biases = math.matrix(math.random([numNeurons], -1, 1));
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

			var batches = [];
			trainingData = _.shuffle(trainingData);
			batches.push(trainingData.slice(0, 2));
			batches.push(trainingData.slice(2));

			var numBatches = batches.length;

			for(var b = 0; b < numBatches; b++) {

				averageError = 0;

				// initialize delta weight variables
				for(var i = 0; i < ann.layers.length; i++) {
					ann.layers[i].m_delta_w = math.zeros(ann.layers[i].m_weights.valueOf().length,
						ann.layers[i].m_weights.valueOf()[0].length);
					ann.layers[i].m_delta_b = math.zeros(ann.layers[i].m_biases.valueOf().length, 1);
				}
				
				_.forEach(batches[b], function (e) {
					// backpropagation
					var err = backProp(ann, e.x, e.y);
					//averageError += math.pow(err,2);
				});

					// update weight matrices
				for(var j = 0; j < ann.layers.length; j++) {
					var m_delta_rw = math.multiply(sink.conf.rate, ann.layers[j].m_delta_w);
					var m_delta_rb = math.multiply(sink.conf.rate, ann.layers[j].m_delta_b);

					ann.layers[j].m_weights = math.subtract(ann.layers[j].m_weights, m_delta_rw);
					ann.layers[j].m_biases = math.subtract(ann.layers[j].m_biases, m_delta_rb);
				}
				console.log(batches[b].length);
				ann.avgErr = averageError / batches[b].length;
			}	

			if(sink.conf.logging === SINK_LOGLEVEL_FULL) {
				console.log('\033[2J');
				console.log('Epoch ' + (ep+1) + ' completed');
				console.log('Average error: ' + ann.avgErr);
			}
		}

		return 1;
	};

	sink.classify = function (ann, inputs) {
		inputs = math.transpose(math.matrix([inputs]));
		var inp = inputs.valueOf();

		for (var i = 0; i < ann.layers.length; i++) {
			var oldInputs = inputs;
			inputs = ann.layers[i].activationFn(math.add(math.multiply(ann.layers[i].m_weights, oldInputs),
				ann.layers[i].m_biases));
		}

		var outp = inputs.valueOf();

		if(sink.conf.logging === SINK_LOGLEVEL_FULL) {
			console.log("Input -> " + math.format(inp, 14));
			console.log("Output -> " + math.format(outp, 14) + "\n");
		}

		return outp;
	};

	sink.testSuite = {};
	sink.testSuite.vectorize = vectorize;
	sink.testSuite.hadamard = hadamard;
	sink.testSuite.costDerivative = costDerivative;
	sink.testSuite.backProp = backProp;

	module.exports = sink;

}.call(this));

