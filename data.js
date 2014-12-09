(function () {

	var _ = require('./lib/lodash');
	var fs = require('fs');
	var csv = require('fast-csv');
	var data = {};

	data.readCsv = function (filename, callback) {
		var result = [];
		var stream = fs.createReadStream(filename);
		var counter = 0;
		var csvStream = csv()
		.on("data", function(data){
			data = _.map(data, function (d) {
				return parseInt(d);
			});
			result.push(data);
		})
		.on("end", function(){
			console.log("done");
			callback(result);
		});

		stream.pipe(csvStream);
	};

	module.exports = data;

}.call(this));

