var knn = require('../knn');
var data = require('../common/data');
var _ = require('../lib/lodash');

var testLabels = [];

var trainingData = data.readCsv('/Users/karlpetersson/ML/examples/lol.csv');
var trainingExamples = trainingData.map(function (line) {
    return {
        x: _.initial(line).map(function (v) { return parseFloat(v);}),
        y: parseInt(_.last(line))
    }
});

//var testData = data.readCsv('data.csv');

trainingExamples = _.shuffle(trainingExamples);

var hej = knn.init(trainingExamples);

var test = [];
test.push(parseFloat(process.argv[2]));
test.push(parseFloat(process.argv[3]));
test.push(parseFloat(process.argv[4]));

var guess = knn.classify(hej, test, 10);

console.log(guess);

//console.log("hej");

//console.log(((totcorrect/totguesses) * 100).toFixed(2) + "% (" + totcorrect + "/" + totguesses + ") correct guesses");
