$(document).ready(function() {

    var size = 533;
    var step = (size-1) / 28;
    var PencilEraserIter = 1;
    var PencilEraserSizes = [0.5 * step, step, 1.5 * step, 2 * step, 2.5 * step];
    var PencilEraserNames = ["Small", "Normal", "Large", "Extra Large", "Huge"];

    $("#main").css({
        "height": size + "px",
        "min-height": size + "px",
        "max-height": size + "px",
        "width": size + "px",
        "min-width": size + "px",
        "max-width": size + "px"
    });

    var img_size = 260;

    var draw = new Draw(size, step);
    draw.initialize();
    
    $('#clearBTN').on("click", function() { 
        draw.initialize();
        $("#results_table").empty();
        $("#wordcloud").empty();
    });

    $('#smallerBTN').on("click", function() {
        PencilEraserIter = PencilEraserIter != 0 ? --PencilEraserIter : 0;
        $("#pencis_eraser_size").text(PencilEraserNames[PencilEraserIter]);
        draw.changeSize(PencilEraserSizes[PencilEraserIter])
    });

    $('#largerBTN').on("click", function() {
        PencilEraserIter = PencilEraserIter != PencilEraserNames.length-1 ? ++PencilEraserIter : PencilEraserNames.length-1;
        $("#pencis_eraser_size").text(PencilEraserNames[PencilEraserIter]);
        draw.changeSize(PencilEraserSizes[PencilEraserIter])
    });

    $('#submitBTN').on("click", function() {
        draw.drawInput(img_size);
    });
});


var Draw = function(size, step) {
    var self = this;

    self.radius = step;
    self.drawing = false;
    self.prev = {x: 0, y: 0};

    self.canvas = document.getElementById('main');
    self.img = document.getElementById('img');
    self.ctx = self.canvas.getContext('2d');
    self.canvas.width = size;
    self.canvas.height = size;

    self.canvas.addEventListener('mousedown', function(e) {
        self.canvas.style.cursor = 'default';
        self.drawing = true;
        self.prev = self.getPosition(e.clientX, e.clientY);
    }, false);
    self.canvas.addEventListener('mouseup', function(e) {
        self.drawing = false;
    }, false);
    self.canvas.addEventListener('mouseout', function(e) {
        self.drawing = false;
    }, false);
    self.canvas.addEventListener('mousemove', function(e) {
        if (self.drawing) {
            var curr = self.getPosition(e.clientX, e.clientY);
            self.ctx.lineWidth = self.radius;
            self.ctx.lineCap = 'round';
            self.ctx.beginPath();
            self.ctx.moveTo(self.prev.x, self.prev.y);
            self.ctx.lineTo(curr.x, curr.y);
            self.ctx.stroke();
            self.ctx.closePath();
            self.prev = curr;
        }
    }, false);

    this.initialize = function() {
        self.ctx.fillStyle = '#FFFFFF';
        self.ctx.fillRect(0, 0, self.canvas.width, self.canvas.height);
        self.ctx.lineWidth = 1;
        self.ctx.strokeRect(0, 0, self.canvas.width, self.canvas.height);
        self.ctx.lineWidth = 0.05;
        for (var i = 0; i < 27; i++) {
            self.ctx.beginPath();
            self.ctx.moveTo((i + 1) * step,   0);
            self.ctx.lineTo((i + 1) * step, self.canvas.height);
            self.ctx.closePath();
            self.ctx.stroke();

            self.ctx.beginPath();
            self.ctx.moveTo(  0, (i + 1) * step);
            self.ctx.lineTo(self.canvas.width, (i + 1) * step);
            self.ctx.closePath();
            self.ctx.stroke();
        }
        self.img.getContext('2d').clearRect(0, 0, 1000, 1000);
    };

    this.getPosition = function(clientX, clientY) {
        var rect = self.canvas.getBoundingClientRect();
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    };

    this.changeSize = function (radius) {
        self.radius = radius;
    };

    this.drawInput = function(size) {
        var w = Math.floor(size / 28);
        var ctx = self.img.getContext('2d');
        var img = new Image();
        var inputs = [];
        var DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
        img.onload = function() {
            var small = document.createElement('canvas').getContext('2d');
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
            var data = small.getImageData(0, 0, 28, 28).data;
            for (var i = 0; i < 28; i++) {
                for (var j = 0; j < 28; j++) {
                    var n = 4 * (i * 28 + j);
                    inputs[i * 28 + j] = (data[n] + data[n + 1] + data[n + 2]) / 3;
                    ctx.fillStyle = 'rgb(' + [data[n], data[n + 1], data[n + 2]].join(',') + ')';
                    ctx.fillRect(j * w, i * w, w, w);
                }
            }
            if (Math.min(inputs) === 255) {
                return;
            }
            $.ajax({
                url: "/mnist",
                type: "POST",
                data: JSON.stringify(inputs),
                contentType: 'application/json;charset=UTF-8',
                success: function (data) {
                    var results = data.output;
                    var best = d3.max(results);
                    var bestArr = results.filter(function(d) { return Math.abs(d - best) <= 20; });
                    $("#results_table").empty();
                    $("#wordcloud").empty();
                    if ($.type(data.output) === "array") {
                        var table = d3.select("#results_table").append("table")
                            .attr("class", "table table-striped table-bordered table-condensed");
                        table.append("thead").append("tr")
                            .selectAll("th")
                            .data(["Digit", "Accuracy in %"]).enter()
                            .append("th")
                            .text(function(column) { return column; });
                        var rows = table.append("tbody").selectAll("tr")
                            .data(results).enter()
                            .append("tr")
                            .classed("success", function(d) { return $.inArray(d, bestArr) !== -1; })
                            .style("font-weight", function(d) { return $.inArray(d, bestArr) !== -1 ? "bold" : "normal"})
                            .style({ "line-height": "5", "min-height": "5", "height": "5", "max-height": "5", "padding": "0px" });
                        rows.selectAll("td")
                            .data(function(row, i) { return [i, row]; })
                            .enter()
                            .append("td")
                            .style({ "line-height": "1.6", "min-height": "5", "height": "5", "max-height": "5", "padding": "0px" })
                            .html(function(d) { return d; });

                        // Draw wordcloud
                        var word_cloud = new WordCloud().selector("#wordcloud");
                        word_cloud.init();
                        var words = [];
                        var indices = [];
                        for (var i=0; i<results.length; i++) {
                            var val = results[i];
                            words.push({'text': DIGIT_NAMES[i], 'size': 1.75 * (val >= 10 ? val : 10) });
                            if ($.inArray(val, bestArr) !== -1) { 
                                indices.push(i);
                            }
                        }
                        word_cloud.update(words, indices)
                    } else {
                        $("#results_table").html('<h3 class="text-danger" style="font-weight:bold">' + results + '</h3>')
                    }
                }
            })
        };
        img.src = self.canvas.toDataURL();
    }
};


var WordCloud = function() {
    var self = this;
    var colors = d3.scale.category10();
    
    this.selector = function (s) {
        self._selector = s;
        return self;
    };

    this.init = function () {
        self.width = parseInt($(self._selector).css('width'));
        self.height = 330;

        self.svg = d3.select(self._selector).append("svg")
            .attr("width", self.width)
            .attr("height", self.height)
            .append("g")
            .attr("transform", "translate(" + self.width / 2 + "," + self.height / 2 + ")");
    };

    this.update = function(words, indices) {
        d3.layout.cloud()
            .size([self.width, self.height])
            .words(words)
            .padding(5)
            .rotate(function(d,i) { return $.inArray(i, indices) !== -1 ? 0 : (~~(Math.random() * 4) - 2) * 30; })
            .fontSize(function(d) { return d.size; })
            .on("end", draw)
            .start();
    };

    function draw(words) {
        var cloud = self.svg.selectAll("g text")
            .data(words, function(d) { return d.text; });
        // Remove existing words
        cloud.transition().duration(500)
            .style('fill-opacity', 0)
            .attr('font-size', 0)
            .remove();

        cloud.enter().append("text")
            .style("fill", function(d, i) { return colors(i); })
            .attr("text-anchor", "middle")
            .attr('font-size', 1)
            .text(function(d) { return d.text; });

        cloud.transition().duration(1000)
            .style("font-size", function(d) { return d.size + "px"; })
            .attr("transform", function(d) {
                return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
            })
            .style("fill-opacity", 1);
    }
};