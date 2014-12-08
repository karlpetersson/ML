var gd = require('./gd');
var _ = require('./lodash');
var plotly = require('plotly')('kojz','mvp19qfxjf');
var X = [[1, 1], [1, 2], [1, 3], [1, 5], [1, 4], [1, 6], [1, 8]];
var y = [1, 3, 4, 5, 6, 7, 9];
var theta = gd.gradientDescent(X, y);
var J = gd.costFn(X, y, theta);

console.log("theta: "+theta);
console.log("J: "+J);

console.log(gd.xValues(X));

//console.log(gd.normalEquationGradientDescent(X, y));
var xValues = gd.xValues(X);
