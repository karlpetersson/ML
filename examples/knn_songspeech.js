var knn = require('../knn');
var data = require('../common/data');
var _ = require('../lib/lodash');

var testLabels = [];

var trainingData = data.readCsv('lol.csv');
var trainingExamples = trainingData.map(function (line) {
    return {
        //x: _.initial(line).map(function (v) { return parseInt(v, 10);}),
        x: [parseFloat(line[0]), parseFloat(line[1]), parseFloat(line[2])],
        y: parseInt(_.last(line))
    }
});

//var testData = data.readCsv('data.csv');

trainingExamples = _.shuffle(trainingExamples);

var numEx = trainingExamples.length;

console.log(numEx);

var end = ~~(numEx * 0.25);

var testy = trainingExamples.slice(0, end);
trainingExamples = trainingExamples.slice(end);

var testExamples = testy.map(function (d) {
    testLabels.push(d.y);
    return d.x;
});

console.log(_.last(testExamples));

var hej = knn.init(trainingExamples);

var totguesses = 0;
var totcorrect = 0;

for(var i = 0; i < testExamples.length; i++) {
    var guess = knn.classify(hej, testExamples[i], 10);

    //console.log("Example: " + i + ", guess: " + guess);

    totguesses++;
    if(guess == testLabels[i]) {
        totcorrect++;
    }
}

console.log(((totcorrect/totguesses) * 100).toFixed(2) + "% (" + totcorrect + "/" + totguesses + ") correct guesses");
