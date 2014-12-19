var nba = require('../nba');
var data = require('../common/data');
var _ = require('../lib/lodash');

var hej = nba.init();

var trainingData = data.readCsv('optdigits.tra');
var trainingDataByClass = [];
var trainingExamples = trainingData.map(function (line) {
    var y = parseInt(_.last(line));
    var x = _.initial(line).map(function (v) { 
        return parseInt(v, 10)/50;
    });

    if(!trainingDataByClass[y]) {
        trainingDataByClass[y] = [];
    }

    trainingDataByClass[y].push(x);
});

var testData = data.readCsv('optdigits.tes');
testExamples = testData.map(function (line) {
    var y = parseInt(_.last(line));
    var x = _.initial(line).map(function (v) { 
        return parseInt(v, 10)/50;
    });
    return {x: x, y: y};
});

nba.train(hej, trainingDataByClass);

var totguesses = 0;
var totcorrect = 0;

for(var i = 0; i < testExamples.length; i++) {
    var guess = nba.predict(hej, testExamples[i].x);
    totguesses++;

    var idx = 0;
    var num = -1.0;

    for(var lol = 0; lol < guess.length; lol++) {
        if(parseFloat(guess[lol]) > parseFloat(num)) {
            num = guess[lol];
            idx = lol;
        }
    }

    if(idx == testExamples[i].y) {
        totcorrect++;
    }

}
console.log(((totcorrect/totguesses) * 100).toFixed(2) + "% (" + totcorrect + "/" + totguesses + ") correct guesses");
