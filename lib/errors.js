(function () {
    var util = require('util');
    var err = {};

    function $errors(errors) {
        function GenericError(msg, constructor) {
            Error.captureStackTrace(this, constructor || this);
            this.message = msg || 'Unspecified error message';
        }

        function createHandler(msg) {
            return function () {
                errorFn.super_.call(this, msg, this.constructor);
            };
        }

        util.inherits(GenericError, Error);
        GenericError.prototype.name = 'Generic Error';

        var results = {};

        for(var errorName in errors) {
            var errorFn = results[errorName] = createHandler(errors[errorName]);
            util.inherits(errorFn, GenericError);
            errorFn.prototype.name = errorName;
        }

        return results;

    }

    module.exports = $errors;

}.call(this));
