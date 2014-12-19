var sink = require('../ann');
var data = require('../common/data');
var _ = require('../lib/lodash');

var hej = sink.init([64,64,10], [[sink.sigmoid, sink.sigmoidPrime], [sink.sigmoid, sink.sigmoidPrime]]);

sink.conf.rate = 0.1;
sink.conf.logging = 1;

var trainingData = data.readCsv('optdigits.tra');
var trainingExamples = trainingData.map(function (line) {
    var y = [0,0,0,0,0,0,0,0,0,0];
    var x = _.initial(line).map(function (v) {
        return parseInt(v)/50;
    });
    y[parseInt(_.last(line))] = 1;
    return {x: x, y: y};
});

var testData = data.readCsv('optdigits.tes');
var testExamples = testData.map(function (line) {
    var y = parseInt(_.last(line));
    var x = _.initial(line).map(function (v) {
        return parseInt(v)/50;
    });
    return {x: x, y: y};
});

for(var i = 0; i < 5; i++) {
    var now = Date.now();
    sink.shuffle(trainingExamples);
    sink.train(hej, trainingExamples);
    var then = Date.now();
    console.log('Epoch completed');
    console.log("time spent: " + (then - now));
}

var numtest = testExamples.length;
var totguesses = 0;
var totcorrect = 0;
for(var i = 0; i < numtest; i++) {
    totguesses++;

    var guess = sink.predict(hej, testExamples[i].x);

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
