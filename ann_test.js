var sink = require('./ann');
var math = require('./lib/math');
var _ = require('./lib/lodash');

function testHadamard () {
	var result = 1;
	var v1 = math.matrix([[2],[3],[2],[3]]);
	var v2 = math.matrix([[3],[2],[3],[2]]);
	var v3 = math.matrix([[1],[1]]);
	var fourProduct = sink.testSuite.hadamard(v1, v2);
	var twoFourProduct = sink.testSuite.hadamard(v1, v3);
	result &= _.isEqual(fourProduct._data, [[6],[6],[6],[6]]);
	result &= _.isEqual(twoFourProduct._data, []);
	return result;
}

console.log("Hamard product -> " + testHadamard());