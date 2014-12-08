var sink = require('./ann');
var data = require('./data');
var math = require('./lib/math');
var _ = require('./lib/lodash');
//ann.set(hej, 3, 1, 2);

//var hej = sink.init(2,1,2,1);
var hej = sink.init([64,64,10], [[sink.sigmoid, sink.sigmoidPrime]]);
//console.log(ann.feedforward(hej, [1, 1]));
//sink.conf.rate = 0.3;
sink.conf.logging = 0;

var trainingData;
var testData;
data.readCsv('optdigits.tra', function (result) {
	trainingData = _.map(result, function(line) {
		
		xvalues = _.map(_.initial(line), function (d) {
			return d/50;
		});
		//console.log(line);
		var y = [0,0,0,0,0,0,0,0,0,0];
		var digit = _.last(line);
		y[digit] = 1;
		return {x: xvalues, y: y};
	});

	data.readCsv('optdigits.tes', function (res) {
		testData = _.map(res, function(line) {
			xvalues = _.map(_.initial(line), function (d) {
				return d/50;
			});
			return {x: xvalues, y: _.last(line)};
		});	

		//console.log(trainingData[0]);
		//console.log(testData[0]);

		console.log(trainingData.length);

		sink.train(hej, trainingData, 1);

		var numtest = testData.length;
		var totguesses = 0;
		var totcorrect = 0;
		for(var i = 0; i < numtest; i++) {
			totguesses++;
			var guess = sink.classify(hej, testData[i].x);
			if(guess[testData[i].y] > 0.9) {
				totcorrect++;
			}
		}
		console.log(totcorrect + "/" + totguesses + " correct guesses");
	});
});




//sink.conf.logging = 0;

//while(hej.avgErr > 0.002) {
//sink.train(hej, trainingData, 500);
//
/*
sink.classify(hej, [1,0]);
sink.classify(hej, [0,1]);
sink.classify(hej, [1,1]);
sink.classify(hej, [0,0]);*/


//console.log(hej.layers[0].weightMatrix);
//console.log(hej.layers[1].weightMatrix);