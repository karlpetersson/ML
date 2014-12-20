(function () {

    var randomForest = {};

    function Node () {

    }

    function negativeLogSum (vec) {
        var sum = 0;
        for(var i = 0, len = vec.length; i < len; i++) {
            sum -= vec[i] * Math.log(vec[i]);
        }
    }

    function DecisionTree (set, labels) {  
        var labels = {},
            len = set.length,
            numFeatures = set[0].length;

        for(var i = 0; i < len; i++) {
            if(!proportions[labels[i]]) {
                proportions[labels[i]] = 0;
            }
            proportions[labels[i]] += 1;
        }

        var entropy = 0;
        for(p in proportions) {
            proportions[p] /= len;
            entropy -= proportions[p] * Math.log(proportions[p]);
        }

        


    }

    module.exports = randomForest;

}.call(this));

