
//#region Global Variable
//Global variable
var speed = 1000;
var x = 0;
var continueSwapping = true;

var images = [];

function getImages(){
	var tables = $('table');
	console.log(tables)
	tables.map(function(item){
	     var src = $(this).find("img").attr("src");
		images.push(src); 
		console.log("the image is",$(this).find("img").attr("src"))
	})
}

getImages();

changeImage();
//For test_case
var FadeOut_btn, Change_Image, ToggleToolTip, PlayStop_Btn, StopText, PlayText;
//endregion Global Variable

function InitializeVariable(list, fadeOut = "FadeOut_btn", changeImage = "Change_Image", toggleToolTip = '[data-toggle="tooltip"]',
							playStop = "PlayStop_Btn", stopText = "Stop", playText = "Play")
{
	images = list;
	FadeOut_btn = "FadeOut_btn";
	Change_Image = "Change_Image";
	ToggleToolTip = '[data-toggle="tooltip"]';
	PlayStop_Btn = "PlayStop_Btn";
	StopText = "Stop";
	PlayText = "Play";
}

//TODO: integrate make it useable
$(document).ready(function(){
	$(FadeOut_btn).click(function(){
		$(Change_Image).fadeOut("fast");
	});
});

function fadeImg(val, fade)
{
	
	if(fade === true)
	{
		val--;
	}/*else
	{
		val ++;
	}*/
	
	if(val <= 0)
	{
		fade = false;
	}
	if(val > 0 && val < 100)
	{
		el = document.getElementById(Change_Image);
		el.style.opacity = val / 100;
		setTimeout(function(){fadeImg(val, fade);}, 10);
		//changeImage();
	}
}
//END TODO

$(document).ready(function(){
	$(ToggleToolTip).tooltip();
});

function getCurrentDate()
{
	var d = new Date();
	d.getTime();
}

function updateImageListWithList(list)
{
	images = list;
}

function updateImageListWithElement(elem)
{
	images.push(elem);
}

function requestChangeImage()
{
	if (document.getElementById(PlayStop_Btn).textContent == StopText)
	{
		changeImage();
	}
}

function changeImage(){
			console.log("swap")
	if(continueSwapping){
		var d = new Date();
		$("#Change_Image").attr('src', images[x] +"?"+d.getTime());
		x++;
		if(x >= images.length)
		{
			x = 0;
		} 
		setTimeout("changeImage()", speed);
	}
}
$("#Change_Image").click(function(e){
	continueSwapping = !continueSwapping;
	console.log("cahnge swap",continueSwapping)
	changeImage();
})

function changePlayStopButtonState() 
{
	var elem = document.getElementById(PlayStop_Btn);
	if (elem.textContent == PlayText)
	{
		elem.textContent = StopText;
		$(document).ready(requestChangeImage);
	}
	else 
	{
		elem.textContent = PlayText;
		clearTimeout(continueSwapping);
	}
}

function changeSpeed(speedChange)
{
	speed = speedChange;
	clearTimeout(continueSwapping);
	$(document).ready(requestChangeImage);
}

function confirmSpeed(speedChange)
{
	speed = document.getElementById(speedChange).value;
	clearTimeout(continueSwapping);
	$(document).ready(requestChangeImage);
}
