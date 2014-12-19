var sink = require('../ann');
var data = require('../data');
var _ = require('../lib/lodash');

var hej = sink.init([64,64,10], [[sink.sigmoid, sink.sigmoidPrime], [sink.sigmoid, sink.sigmoidPrime]]);

sink.conf.rate = 0.1;
sink.conf.logging = 1;

var trainingData;
var testData;

data.readCsv('optdigits.tra', function (result) {
    trainingData = _.map(result, function(line) {
        xvalues = _.map(_.initial(line), function (d) {
            return d/50;
        });
        var y = [0,0,0,0,0,0,0,0,0,0];
        y[_.last(line)] = 1;
        return {x: xvalues, y: y};
    });

    data.readCsv('optdigits.tes', function (res) {
        testData = _.map(res, function(line) {
            xvalues = _.map(_.initial(line), function (d) {
                return d/50;
            });
            return {x: xvalues, y: _.last(line)};
        }); 

        for(var i = 0; i < 5; i++) {
            var now = Date.now();
            sink.shuffle(trainingData);
            sink.train(hej, trainingData);
            var then = Date.now();
            console.log('Epoch completed');
            console.log("time spent: " + (then - now));
        }

        var numtest = testData.length;
        var totguesses = 0;
        var totcorrect = 0;
        for(var i = 0; i < numtest; i++) {
            totguesses++;

            var guess = sink.predict(hej, testData[i].x);

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
