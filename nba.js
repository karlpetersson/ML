(function () {
	var _ = require('./lib/lodash');
	var $m = require('./fastmath');
	var $errors = require('./lib/errors');

	var _nba = {};

	var SQUARE_ROOT_2PI = Math.sqrt(2 * Math.PI);
	var EPSILON = 0.02;

	var ERRORS = $errors({
		DIM_ERR: 'Wrong input dimension for classifier',
		DIM_MISMATCH: 'Dimension mismatch between training examples'
	});

	function Nba () {
		this.featureSummary = [];
		this.priors = [];
	}

	function init () {
		return new Nba();
	}

	function summarize (set) {
		var len = set.length,
			numFeatures = set[0].length,
			features = [];

		for(var f = 0; f < numFeatures; f++) {
			var fsum = [];
			for(var t = 0; t < len; t++) {
				if(set[t][f] === undefined) {
					throw new ERRORS.DIM_MISMATCH();
				}
				fsum[t] = set[t][f];
			}

			features[f] = [$m.mean(fsum), $m.std(fsum) + EPSILON];
		}

		return features;
	}

	function calcProbability (x, mean, stdev) {
		return (1 / (SQUARE_ROOT_2PI * stdev)) *
			Math.exp(-(Math.pow(x - mean, 2) / (2 * Math.pow(stdev, 2))));
	}

	function train (nba, sets) {
		var len = sets.length,
			total = 0;

		for(var i = 0; i < len; i++) {
			total += sets[i].length;
		}

		for(var j = 0; j < len; j++) {
			nba.featureSummary[j] = summarize(sets[j]);
			nba.priors[j] = sets[j].length / total;
		}

	}

	function predict (nba, example) {
		var len = example.length,
			numClasses = nba.featureSummary.length,
			pClasses = [];

		for(var c = 0; c < numClasses; c++) {
			if(len !== nba.featureSummary[c].length) {
				throw new ERRORS.DIM_ERR();
			}
			pClasses[c] = nba.priors[c];
			for(var f = 0; f < len; f++) {
				pClasses[c] *= calcProbability(example[f], nba.featureSummary[c][f][0], nba.featureSummary[c][f][1]);
			}
		}

		return pClasses;

	}

	_nba.predict = predict;
	_nba.train = train;
	_nba.calcProbability = calcProbability;
	_nba.summarize = summarize;
	_nba.init = init;

	module.exports = _nba;

}.call(this));

