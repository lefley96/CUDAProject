var fs = require('fs');

var samples = process.argv[2] || 100;
var channels = process.argv[3] || 4;
var stream = fs.createWriteStream('input.txt', { flags: 'w' });

for (var i = 0; i < samples; i++) {
	for (var j = 0; j < channels; j++) {
		stream.write(Math.floor(Math.random() * 100) + (j != channels - 1 ? ';' : '\n'));
	}
}

stream.end();
