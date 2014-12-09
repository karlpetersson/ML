var m = require('./fastmath');
var math = require('./lib/math')

var mat = [[1,2,3], [1,2,3]];
var mat2 = [[2,2], [2,2], [2,2]];
var vec1 = [[1,2,3]];
var vec2 = [[2],[3],[4]];

function ok (a) {
	return a + 1;
}

console.log("matrices");
console.log(mat);
console.log(mat2);
console.log(vec1);
console.log("--------");
console.log("transpose mat1");
var lol = m.transpose(mat);
console.log(m.transpose(lol));
console.log(m.transpose(mat));
console.log("--------");
console.log("zeros 3 rows 2 columns");
console.log(m.zeros(3,3));
console.log("--------");
console.log("mat1 * mat2");
console.log(m.multMatrixMatrix(mat, mat2));
console.log("--------");
console.log("mat2 * mat1");
console.log(m.multMatrixMatrix(mat2, mat));
console.log("--------");
console.log("subtract mat1 mat1");
console.log(m.subtractMatrixMatrix(mat, mat));
console.log("--------");
console.log("add mat1 mat1");
console.log(m.addMatrixMatrix(vec2, vec2));
console.log("--------");
console.log("multiply mat1 scalar 2");
console.log(m.multMatrixScalar(mat, 2));
console.log("--------");
console.log("multiply mat3 mat2");
console.log(m.multMatrixMatrix(vec1, mat2));
console.log("--------");
console.log("multiply mat3(T) mat3(T) elementwise");
console.log(m.multMatrixElementwise(vec1, vec1));
console.log("--------");
console.log("multiply mat2 mat2 elementwise");
console.log(m.multMatrixElementwise(mat2, mat2));
console.log("--------");
console.log("random");
console.log(m.random(3, 3));
console.log("--------");
console.log("vectorize mat3(T) (a + 1)");
var vectorized = m.vectorize(ok);
console.log(vectorized(m.transpose(vec1)));
console.log("--------");