var ann = require('../ann');
var data = require('../data');

var testann = ann.init([2,2,1], [[ann.tanh, ann.tanhPrime]]);

var trainingData = [
	{x: [1, 0], y:[1] },
	{x: [0, 1], y:[1] },
	{x: [1, 1], y:[0] },
	{x: [0, 0], y:[0] }
];

var testData = [
	[1,0],[0,1],[1,1],[0,0]
];

ann.conf.rate = 0.3;

for(var i = 0; i < 500; i++) {
	ann.train(testann, trainingData);
}

for(var j = 0, len = testData.length; j < len; j++) {
	console.log("input -> " + testData[j]);
	console.log("output -> " + ann.predict(testann, testData[j]) + "\n");
}