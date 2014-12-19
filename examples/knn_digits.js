var knn = require('../knn');
var data = require('../data');
var _ = require('../lib/lodash');

var trainingData;
var testData;
var trainingLabels = [];
var testLabels = [];

data.readCsv('optdigits.tra', function (result) {

	trainingData = _.map(result, function(line) {
		return {x: _.initial(line), y: _.last(line)};
	});

	data.readCsv('optdigits.tes', function (res) {
		
		testData = _.map(res, function(line) {
			testLabels.push(_.last(line));
			return _.initial(line);
		});

		var hej = knn.init(trainingData);

		var totguesses = 0;
		var totcorrect = 0;

		for(var i = 0; i < testData.length; i++) {
			var guess = knn.classify(hej, testData[i], 1);

			console.log("Guess: " + i + " prediction: " + guess);

			totguesses++;

			if(guess == testLabels[i]) {
				totcorrect++;
			}
		}
		console.log(((totcorrect/totguesses) * 100).toFixed(2) + "% (" + totcorrect + "/" + totguesses + ") correct guesses");
	});
});
