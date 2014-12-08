var sink = require('./ann');

//ann.set(hej, 3, 1, 2);

//var hej = sink.init(2,1,2,1);
var hej = sink.init([2,2,1], sink.sigmoid, sink.sigmoidPrime);
//console.log(ann.feedforward(hej, [1, 1]));

var batch = [
	{x: [1, 0], y:[1] },
	{x: [0, 1], y:[1] },
	{x: [1, 1], y:[0] },
	{x: [0, 0], y:[0] }
];

/*var lolbatch = [{x:[0,0], y:[0]},
	{x: [1,1], y:[1]}];*/
/*
console.log(hej.layers[0].weightMatrix);
console.log(hej.layers[1].weightMatrix);
console.log("------");*/

while(hej.avgErr > 0.002) {
	sink.train(hej, batch);
}

sink.classify(hej, [1,0]);
sink.classify(hej, [0,1]);
sink.classify(hej, [1,1]);
sink.classify(hej, [0,0]);


//console.log(hej.layers[0].weightMatrix);
//console.log(hej.layers[1].weightMatrix);