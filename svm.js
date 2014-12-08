var math = require('./math');

// COLOR,SIZE,DIP/STRETCH,AGE,INFLATED/DEFLATED

var svm = {};

var Svm = function (c, sigma) {
	this.c = c;
	this.sigma = sigma;
};

var _sum = function (arr) {
	return _.reduce(arr, function (sum, num) {
		return sum + num;
	});
};

var _cost0 = function (z) {

};

var _cost1 = function (z) {

};

sink.initSvm = function (c, sigma) {
	return new Svm(c, sigma);
};

sink.train = function (svm, kernel, X, y) {
	landmarks = X;
	var f_set = X.map(function (value, index, matrix) {
		return landmarks.map(function (iv, ii, im) {
			return kernel(value, iv);
		});
	});
	console.log(f_set);

	// minimize cost

};

sink.cost = function (f, y, theta) {
	var terms = y.map(function (value, index, matrix) {
		return value * _cost1(math.multiply(math.transpose(theta), math.subset(f, index))) +
			(1 - value) * _cost0(math.multiply(math.transpose(theta), math.subset(f, index)));
	});
	return C * _sum(terms) + math.multiply(math.transpose(theta), theta);
};

// gaussian gaussian/similiarity function
sink.gaussianKernel = function (x, l) {
	var eucdist_squared = math.pow(math.norm(math.subtract(x,l)), 2);
	return math.exp(-(eucdist_squared / (2*sigma_squared)));
};

sink.classify = function(svm, x) {
	// input to cost0 or cost1
	// return 1 or 0
};

module.exports = sink;