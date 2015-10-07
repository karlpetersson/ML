var knn = require('../knn');
var data = require('../common/data');
var _ = require('../lib/lodash');

var testLabels = [];

var trainingData = data.readCsv('optdigits.tra');
var trainingExamples = trainingData.map(function (line) {
    return {
        x: _.initial(line).map(function (v) { return parseInt(v, 10);}),
        y: parseInt(_.last(line))
    }
});

var testData = data.readCsv('optdigits.tes');
var testExamples = testData.map(function (line) {
    testLabels.push(parseInt(_.last(line)));
    return _.initial(line).map(function (v) { return parseInt(v, 10);});
});

console.log(_.last(testLabels));

var hej = knn.init(trainingExamples);

var totguesses = 0;
var totcorrect = 0;

for(var i = 0; i < testExamples.length; i++) {
    var guess = knn.classify(hej, testExamples[i], 1);

    //console.log("Example: " + i + ", guess: " + guess);

    totguesses++;
    if(guess == testLabels[i]) {
        totcorrect++;
    }
}

console.log(((totcorrect/totguesses) * 100).toFixed(2) + "% (" + totcorrect + "/" + totguesses + ") correct guesses");
