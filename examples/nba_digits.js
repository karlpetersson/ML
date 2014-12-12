var nba = require('../nba');
var data = require('../data');
var _ = require('../lib/lodash');

var hej = nba.init();

var trainingData;
var testData;

data.readCsv('optdigits.tra', function (result) {
	var trainingDataByClass = [];

	trainingData = _.map(result, function(line) {
		if(!trainingDataByClass[_.last(line)]) {
			trainingDataByClass[_.last(line)] = [];
		}
		xvalues = _.map(_.initial(line), function (d) {
			return d/50;
		});
		trainingDataByClass[_.last(line)].push(xvalues);
	});

	//console.log(trainingDataByClass);

	data.readCsv('optdigits.tes', function (res) {
		
		testData = _.map(res, function(line) {
			xvalues = _.map(_.initial(line), function (d) {
				return d/50;
			});
			return {x: xvalues, y: _.last(line)};
		});

		nba.train(hej, trainingDataByClass);

		var totguesses = 0;
		var totcorrect = 0;

		for(var i = 0; i < testData.length; i++) {
			var guess = nba.predict(hej, testData[i].x);
			totguesses++;

			var idx = 0;
			var num = -1.0;

			for(var lol = 0; lol < guess.length; lol++) {
				if(parseFloat(guess[lol]) > parseFloat(num)) {
					num = guess[lol];
					idx = lol;
				}
			}

			if(idx == testData[i].y) {
				totcorrect++;
			}

		}
		console.log(((totcorrect/totguesses) * 100).toFixed(2) + "% (" + totcorrect + "/" + totguesses + ") correct guesses");
	});
});
