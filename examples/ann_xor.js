var sink = require('../ann');
var data = require('../data');
//ann.set(hej, 3, 1, 2);

//var hej = sink.init(2,1,2,1);
var hej = sink.init([2,2,1], [[sink.tanh, sink.tanhPrime]]);
//console.log(ann.feedforward(hej, [1, 1]));

var trainingData = [
	{x: [1, 0], y:[1] },
	{x: [0, 1], y:[1] },
	{x: [1, 1], y:[0] },
	{x: [0, 0], y:[0] }
];

sink.conf.rate = 0.3;
//sink.conf.logging = 0;

//while(hej.avgErr > 0.002) {
sink.train(hej, trainingData, 500);
//

sink.classify(hej, [1,0]);
sink.classify(hej, [0,1]);
sink.classify(hej, [1,1]);
sink.classify(hej, [0,0]);


//console.log(hej.layers[0].weightMatrix);
//console.log(hej.layers[1].weightMatrix);