var _ = require('./lodash');
var math = require('./math');

var gd = {};

var _add = function (a, b) {
	return a + b;
};

var _sum = function (vec) {
	return _.reduce(vec, function (sum, v) { return _add(sum, v); });
};

var _sqrErr = function (x, y) {
	return x.map(function (value, index, matrix) {
		return Math.pow(value - math.subset(y, math.index(math.squeeze(index))), 2);
	});
};

gd.costFn = function (X, y, theta) {
	if(X[0].length !== theta.length) {
		return console.error("Feature vector theta must match length of training example");
	}

	var m = X.length;
	var m_x = math.matrix(X);
	var m_y = math.matrix(y);
	var m_theta = math.matrix(theta);
	var predictions = math.multiply(m_x, theta);
	predictions = math.squeeze(predictions);
	var squareErrors = _sqrErr(predictions, m_y);
	var J = (1 / (2 * m)) * _sum(squareErrors._data);
	return J;
};

gd.gradientDescent = function (X, y) {
	var n = y.length;
	if(n < 1000) {
		return gd.normalEquation(X, y);
	} else {
		console.log("n large");
	}
};

gd.normalEquation = function (X, y) {
	return math.multiply(math.inv(math.multiply(math.transpose(X), X)), math.multiply(math.transpose(X), y));
};

gd.batchGradientDescent = function (X, y) {

};

gd.xValues = function (X) {
	return X.map(function (value, index, matrix) {
		return value[1];
	});
};

gd.linear = function (X, theta) {
	if(X[0].length !== theta.length) return console.error("Number of features in X does not match length of theta");
	return X.map(function (value, index, matrix) {
		var result = 0;
		for(var i = 0, len = value.length; i < len; i++) {
			result += value[i] * theta[i];
		}
		return result;
	});
};

module.exports = gd;

