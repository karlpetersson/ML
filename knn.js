(function () {

    var $m = require('./common/math');

    var kNearestNeighbours = {};

    function Node(value, label, leftChild, rightChild) {
        this.value = value;
        this.label = label;
        this.leftChild = leftChild;
        this.rightChild = rightChild;
    }

    function getMaxIdx (list, k) {
        var idx = 0,
            max = 0,
            len = list.length;

        if(len < k) {
            list[len] = Infinity;
            return len;
        }

        for(var i = 0; i < len; i++) {
            if(list[i] > max) {
                max = list[i];
                idx = i;
            }
        }

        return idx;
    }

    function kdTree (points, depth) {
        
        if(!points[0]) {
            return null;
        }

        var k = points[0].x.length;
        var axis = depth % k;

        points = points.sort(function (a,b) {
            if(a.x[axis] < b.x[axis]) {
                return -1;
            } else if(a.x[axis] > b.x[axis]) {
                return 1;
            }
            return 0;
        });

        var m = Math.floor(points.length / 2);

        return new Node(
            points[m].x, points[m].y,
            kdTree(points.slice(0, m), depth + 1),
            kdTree(points.slice(m + 1), depth + 1)
        );
    }


    function init (points) {
        return kdTree(points, 0);
    }

    function classify (root, point, k) {
        var bestGuesses = [],
            labels = [];

        // Nearest neighbours search
        (function next (node, depth) {
            if(!node) return;

            var axis = depth % point.length;
            var otherBranch = null;

            if(point[axis] < node.value[axis]) {
                next(node.leftChild, depth + 1);
                otherBranch = node.rightChild;
            } else {
                next(node.rightChild, depth + 1);
                otherBranch = node.leftChild;
            }

            var idx = getMaxIdx(bestGuesses, k);
            var dist = $m.euclidianDistance(node.value, point);

            if (dist < bestGuesses[idx]) {
                bestGuesses[idx] = dist;
                labels[idx] = node.label;
            }

            if (Math.abs(point[axis] - node.value[axis]) < bestGuesses[idx]) {
                next(otherBranch, depth + 1);
            }

            return;

        })(root, 0);

        var votes = {};
        for(var i = 0, len = labels.length; i < len; i++) {
            if(!votes[labels[i]]) {
                votes[labels[i]] = 0;
            }
            votes[labels[i]]++;
        }

        var best = null;
        for(var vote in votes) {
                if(!best || votes[vote] > votes[best]) {
                best = vote;
            }
        }

        return best;
    }

    kNearestNeighbours.kdTree = kdTree;
    kNearestNeighbours.init = init;
    kNearestNeighbours.classify = classify;

    module.exports = kNearestNeighbours;

}.call(this));

