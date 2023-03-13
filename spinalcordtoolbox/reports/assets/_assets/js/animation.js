"use strict";
$("#description").fadeToggle( "fast");
//hide or show description
$("#toggleDescription").click(function() {
    var new_content = $("#toggleDescription").text() == "Hide description" ? "Show description" : "Hide description";
	$("#description").fadeToggle( "fast");
	$("#toggleDescription").text(new_content);
})
//Global variable
var speed = 1000;
var x = 0;
var continueSwapping = true;
var names = [];

var images = [];

function getImages(){
	var spans = $('#togif').find("span");
	console.log("spans",spans)
	spans.map(function(item){
	if( $(this).attr("data-src")){
             var src = $(this).attr("data-src");
             var name =$(this).attr("data-name");
             names.push(name)
             images.push(src);
		 }
	})
}
//change image each second to simulate gif view
function changeImage(){
	if(continueSwapping){
		var d = new Date();
		$("#gif_image").attr('src', images[x] +"?"+d.getTime());
		x++;
		if(x >= images.length){
		    x = 0;
		}
		setTimeout("changeImage()", speed);
	}
}
getImages();

changeImage();

function getCurrentDate(){
	var d = new Date();
	d.getTime();
}

//make pause
$("#gif_image").click(function(e){
	continueSwapping = !continueSwapping;
	changeImage();
})
