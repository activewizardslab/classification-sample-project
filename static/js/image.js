$(document).ready(function() {
$('#loading').hide();

    $('#upload_file_btn').on("click", function() {
        $('#loading').show();
        var form_data = new FormData($('#upload_file')[0]);
        $.ajax({
            type: 'POST',
            url: '/upload',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                console.log(data);
                if(data.error){
                  $('#uploaded_image').html('<h2>' + data.error + '</h2>');
                  $('#loading').hide();
                }
                else{
                  $('#uploaded_image').html('<img src="static/uploads/' + data.filename + '" class="main_image">');
                  draw(data.labels, data.scores);
              }
            }

        });
    });

    $('#link_file_btn').on("click", function() {
        $('#loading').show();
        $.ajax({
            url: "/link",
            type: "POST",
            data: JSON.stringify({"url": $("#image_url").val()}),
            contentType: 'application/json;charset=UTF-8',
            success: function (data) {
                console.log(data);
                if(data.error){
                  $('#uploaded_image').html('<h2>' + data.error + '</h2>');
                  $('#loading').hide();
                }
                else{
                  $('#uploaded_image').html('<img src="' + data.filename + '" class="main_image">');

                  draw(data.labels, data.scores);
                  //draw()
                }
            }
        })
    });

    $('.test-img').on("click", function(){
        $('#loading').show();
        var self = $(this);
        var img = self.attr('id').split("_").pop() + ".jpg";
        console.log(img);
        $.ajax({
            url: "/test",
            type: "GET",
            data: { img: img },
            success: function (data) {
                console.log(data);
                $('#uploaded_image').html('<img src="static/images/' + img + '" class="main_image">');

                draw(data.labels, data.scores);
                //draw()
            }
        });

   
    });
});

function draw (labels, scores) {

    var labels= labels;
    var scores = scores;

    $('#classes').empty();
    var width = $('#classes').width(),
        height = $('#classes').height();
    var margin = {top: 10, right: 10, bottom: 30, left: 20};
	  var color = "lightsteelblue";
    var barH = (height - margin.top - margin.bottom) / (10*2);

    var xScale = d3.scale.linear().domain([0, 1]).range([0, width - margin.left - margin.right]);
    var yScale = d3.scale.linear().domain([0, 10]).range([0, height - margin.top - margin.bottom]);
    var gridScale = d3.scale.ordinal(d3.range(gridN)).range([0, width - margin.left - margin.right]);

    var gridN = 10,
        step = xScale(1) / gridN;

    var svg = d3.select('#classes').append('svg').attr({'width': width,'height': height});

    // Grid
    svg.append('g')
        .attr('id', 'grid')
		.attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
		.selectAll('line')
		.data(d3.range(gridN + 1)).enter()
		.append('line')
		.attr({
            'x1': function(d, i){ return i * step; },
            'y1': 0,
            'x2': function(d, i){ return i * step; },
			'y2': height - margin.bottom
        })
		.style({ 'stroke': '#e6e6e6', 'stroke-width': '1px' });
    // Top line
    svg.append('line')
		.attr({ 'x1': margin.left, 'y1': margin.top, 'x2': width - margin.right, 'y2': margin.top })
		.style({ 'stroke': '#adadad', 'stroke-width': '1px' });
    // Bottom line
    svg.append('line')
		.attr({ 'x1': margin.left, 'y1': height - margin.bottom + margin.top, 'x2': width - margin.right, 'y2': height - margin.bottom + margin.top })
		.style({ 'stroke': '#adadad', 'stroke-width': '1px' });
    // Left line
    svg.append('line')
		.attr({ 'x1': margin.left, 'y1': margin.top, 'x2': margin.left, 'y2': height + margin.top - margin.bottom })
		.style({ 'stroke': '#adadad', 'stroke-width': '1px' });
    // Right line
    svg.append('line')
		.attr({ 'x1': width - margin.right, 'y1': margin.top, 'x2': width - margin.right, 'y2': height + margin.top - margin.bottom })
        .style({ 'stroke': '#adadad', 'stroke-width': '1px' });
    // x axis
    var	xAxis = d3.svg.axis();
    xAxis.orient('bottom').scale(xScale).tickValues(d3.range(0, 1.1, 0.1));
    svg.append('g')
        .attr("transform", "translate(" + margin.left + "," + (height - margin.bottom + margin.top) + ")")
		.attr('id', 'xAxis')
        .style("font-weight", "bold")
        .call(xAxis);

    // Bars
    var chart = svg.append('g')
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr('id', 'bars')
		.selectAll('rect')
		.data(scores).enter()
        .append('rect')
		.attr('height', barH)
		.attr({ 'x': 0, 'y': function(d, i) { return yScale(i) + barH; } })
        .style({'fill': color, 'opacity': 0.8})
		.attr('width', function(d) { return 0; })
        .transition().duration(1000)
        .attr("width", function(d) {return xScale(d); });

    // Labels
	d3.select('#bars')
        .selectAll('.label')
		.data(labels).enter()
        .append('text')
        .classed("label", true)
		.attr({
            'x': 0,
            'y': function(d, i) { return yScale(i) + 0.75*barH; },
            'dx': 5
        }).attr("text-anchor", "begin")
        .style({ 'fill': '#000', 'font-size': '14px', 'font-weight': 'bold' })
        .text(function(d) { return d; });

    // Scores
	d3.select('#bars')
        .selectAll('.score')
		.data(scores).enter()
        .append('text')
        .classed("score", true)
        .transition().delay(850)
		.attr({
            'x': function(d) {return xScale(d); },
            'y': function(d, i) { return yScale(i) + 1.75*barH; },
            'dx': function(d) { return d >= 0.1 ? -2.5 : 2.5}
        }).text(function(d) { return d; })
        .attr("text-anchor", function(d) { return d >= 0.1 ? "end" : "begin" })
        .style({ 'fill': '#404040', 'font-size': '14px', 'font-weight': 'bold' });
   $('#loading').hide();
}
