(function () {

	var m = {};

	m.multMatrixElementwise = function (m1, m2) {
		var res = [],
			rows = m1.length,
			cols = m1[0].length;

		for (var r = 0; r < rows; r++) {
			res[r] = [];
			for (var c = 0; c < cols; c++) {
				res[r][c] = m1[r][c] * m2[r][c];
			}
		}

		return res;
	};

	m.transpose = function (m) {
		var res = [],
			rows = m.length,
			cols = m[0].length;

		for (var c = 0; c < cols; c++) {
			res[c] = [];
			for (var r = 0; r < rows; r++) {
				res[c][r] = m[r][c];
			}
		}
		return res;
	};

	m.multMatrixScalar = function (m1, scalar) {
		var res = [],
			rows = m1.length,
			cols = m1[0].length;

		for (var r = 0; r < rows; r++) {
			res[r] = [];
			for(var c = 0; c < cols; c++) {
				res[r][c] = m1[r][c] * scalar;
			}
		}
		return res;
	};

	m.subtractMatrixMatrix = function (m1, m2) {
		var res = [],
			rows = m1.length,
			cols = m1[0].length;

		for (var r = 0; r < rows; r++) {
			res[r] = [];
			for(var c = 0; c < cols; c++) {
				res[r][c] = m1[r][c] - m2[r][c];
			}
		}
		return res;
	};

	m.addMatrixMatrix = function (m1, m2) {
		var res = [],
			rows = m1.length,
			cols = m1[0].length;

		for (var r = 0; r < rows; r++) {
			res[r] = [];
			for(var c = 0; c < cols; c++) {
				res[r][c] = m1[r][c] + m2[r][c];
			}
		}
		return res;
	};

	m.multMatrixMatrix = function (m1, m2) {
		var res = [],
			rows = m1.length,
			cols = m2[0].length,
			num = m1[0].length;

		for (var r = 0; r < rows; r++) {
			res[r] = [];
			for (var c = 0; c < cols; c++) {
				var result = 0;
				for (var n = 0; n < num; n++) {
					var p = m1[r][n] * m2[n][c];
					result += p;
				}
				res[r][c] = result;
			}
		}
		return res;
	};

	m.multVectorMatrix = function (v, m) {
		var res = [],
			rows = m.length,
			cols = m[0].length;

		for (var c = 0; c < cols; c++) {
			var result = 0;
			for (var r = 0; r < rows; r++) {
				result += v[r] * m[r][c];
				console.log(v[r]);
			}
			res[c] = result;
		}
		return res;
	};

	m.multMatrixVector = function (m, v) {
		var res = [],
			rows = m.length,
			cols = m[0].length;

		for (var r = 0; r < rows; r++) {
			var result = 0;
			for (var c = 0; c < cols; c++) {
				result += m[r][c] * v[c];
			}
			res[r] = result;
		}
		return res;
	};

	m.zeros = function(rows, cols) {
		var res = [];
		for(var r = 0; r < rows; r++) {
			res[r] = [];
			for(var c = 0; c < cols; c++) {
				res[r][c] = 0;
			}
		}
		return res;
	};

	m.random = function(rows, cols) {
		var res = [];
		for(var r = 0; r < rows; r++) {
			res[r] = [];
			for(var c = 0; c < cols; c++) {
				res[r][c] = (Math.random()*2-1)+(Math.random()*2-1)+(Math.random()*2-1);
			}
		}
		return res;
	};

	m.vectorize = function (fn) {
		return function (vec) {
			var len = vec.length;
			var res = [];
			for(var r = 0; r < len; r++) {
				res[r] = [fn(vec[r][0])];
			}
			return res;
		};
	};

	m.multMatrixElementwiseMutate = function (m1, m2) {
		var rows = m1.length,
			cols = m1[0].length;

		for (var r = 0; r < rows; r++) {
			for (var c = 0; c < cols; c++) {
				m1[r][c] *= m2[r][c];
			}
		}

		return m1;
	};

	m.multMatrixScalarMutate = function (m1, scalar) {
		var rows = m1.length,
			cols = m1[0].length;

		for (var r = 0; r < rows; r++) {
			for(var c = 0; c < cols; c++) {
				m1[r][c] *= scalar;
			}
		}
		return m1;
	};

	m.subtractMatrixMatrixMutate = function (m1, m2) {
		var rows = m1.length,
			cols = m1[0].length;

		for (var r = 0; r < rows; r++) {
			for(var c = 0; c < cols; c++) {
				m1[r][c] -= m2[r][c];
			}
		}
		return m1;
	};

	m.addMatrixMatrixMutate = function (m1, m2) {
		var rows = m1.length,
			cols = m1[0].length;

		for(var r = 0; r < rows; r++) {
			for(var c = 0; c < cols; c++) {
				m1[r][c] += m2[r][c];
			}
		}
		return m1;
	};

	m.sum = function (vec) {
		var sum = 0,
			len = vec.length;
		for(var i = 0; i < len; i++) {
			sum += vec[i];
		}
		return sum;
	};

	m.mean = function (vec) {
		return m.sum(vec)/vec.length;
	};

	m.std = function (vec) {
		var average = m.mean(vec),
			len = vec.length,
			squared = 0;

		for(var i = 0; i < len; i++) {
			squared += Math.pow(vec[i] - average, 2);
		}

		return Math.sqrt(squared/len);
	};

	module.exports = m;

}.call(this));


