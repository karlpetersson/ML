(function () {

    var _ = require('../lib/lodash');
    var fs = require('fs');
    //var csv = require('fast-csv');
    var data = {};

    /*data.readCsv = function (filename, callback) {
        var result = [];
        var stream = fs.createReadStream(filename);
        var counter = 0;
        var csvStream = csv()
        .on("data", function(data){
            data = _.map(data, function (d) {
                return parseInt(d, 10);
            });
            result.push(data);
        })
        .on("end", function(){
            callback(result);
        });

        stream.pipe(csvStream);
    };*/

    data.readCsv = function (filename) {
        var lines = [];

        fs.readFileSync(filename)
        .toString()
        .split('\n')
        .forEach(function (line) {
            if(line.length > 0) {
                lines.push(line.split(","));    
            }
        });

        //console.log(lines[lines.length-1][64]);

        return lines;
    }

    module.exports = data;

}.call(this));

