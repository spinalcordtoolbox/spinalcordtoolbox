/*
//#region Global Variable
//Global variable
var Speed = 1000;
var ImageIndex = 0;
var Images = [];
var ContinueSwapping = false;
var ListImageTmp =  ["images/constraste1_registration_01.png", "images/constraste1_registration_02.png", "images/constraste1_registration_03.png"];

//For test_case
var FadeOut_btn, Change_Image, ToggleToolTip, PopOverToolTip, PlayStop_Btn, StopText, PlayText, ToggleToolTip, PlayStop_Btn, NextImage_Btn, PreviousImage_Btn, ClearImage_Btn, ConfirmImageSelection_Btn, ConfirmImageAdd_Btn, ConfirmSpeed_Btn, StopText, PlayText;
//endregion Global Variable

(function InitializeVariable(list = null, fadeOut_btn = "FadeOut_btn", change_Image = "Change_Image",
	popOverToolTip = '[data-toggle="popover"]', toggleToolTip = '[data-toggle="tooltip"]',	playStop_Btn = "playStop_Btn", 
	nextImage_Btn = "NextImage_Btn",	previousImage_Btn = "PreviousImage_Btn",
	addImage_Btn = "add_Btn", clearImage_Btn = "clear_Btn",	confirmImageSelection_Btn = "confirmImageSelection_Btn", 
	confirmImageAdd_Btn = "confirmImageAdd_Btn", confirmSpeed_Btn = "confirmSpeed_Btn", stopText = "Stop",	playText = "Play")
{
	Images = list;
	FadeOut_btn = fadeOut_btn;
	Change_Image = change_Image;
	ToggleToolTip = toggleToolTip;
	PopOverToolTip = popOverToolTip;
	PlayStop_Btn = playStop_Btn;
	NextImage_Btn = nextImage_Btn;
	PreviousImage_Btn = previousImage_Btn;
	AddImage_Btn = addImage_Btn;
	ClearImage_Btn = clearImage_Btn;
	ConfirmImageSelection_Btn = confirmImageSelection_Btn;
	ConfirmImageAdd_Btn = confirmImageAdd_Btn;
	ConfirmSpeed_Btn = confirmSpeed_Btn;
	StopText = stopText;
	PlayText = playText;
})();

//TODO: integrate make it useable
$(document).ready(function(){
	$(FadeOut_btn).click(function(){
		$(Change_Image).fadeOut("slow");
	});
});

function fadeImg(el, val, fade)
{
	if(fade === true)
	{
		val--;
	}else
	{
		val ++;
	}

	if(val > 0 && val < 100)
	{
		el.style.opacity = val / 100;
		setTimeout(function(){fadeImg(el, val, fade);}, 10);
		changeImage();
	}
}
//END TODO


function getCurrentDate()
{
	var d = new Date();
	d.getTime();
}

function updateImageListWithList(list)
{
	Images = list;
}

function updateImageListWithElement(elem)
{
	Images.push(elem);
}

function requestChangeImage()
{
	if (document.getElementById(PlayStop_Btn).textContent == StopText)
	{
		changeImage();
	}
}

function changeImage()
{
	var d = new Date();
	$("#"+Change_Image).attr('src', Images[ImageIndex] +"?"+d.getTime());
	ImageIndex++;
	if(ImageIndex >= Images.length)
	{
		ImageIndex = 0;
	} 
	ContinueSwapping = setTimeout("changeImage()", Speed);
}


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
		clearTimeout(ContinueSwapping);
	}
}

function changeSpeed(speedChange)
{
	Speed = speedChange;
	clearTimeout(ContinueSwapping);
	$(document).ready(requestChangeImage);
}

function confirmSpeed(speedChange)
{
	Speed = document.getElementById(speedChange).value;
	clearTimeout(ContinueSwapping);
	$(document).ready(requestChangeImage);
}

function selectNextImage()
{
	var d = new Date();
	ImageIndex == Images.length-1 ? ImageIndex = 0 : ImageIndex++;
	console.log(ImageIndex);
	clearTimeout(ContinueSwapping);
	$("#"+Change_Image).attr('src', Images[ImageIndex] +"?"+d.getTime());
}

function selectPreviousImage()
{
	var d = new Date();
	ImageIndex == 0? ImageIndex = Images.length-1 : ImageIndex--;
	console.log(ImageIndex);
	clearTimeout(ContinueSwapping);
	$("#"+Change_Image).attr('src', Images[ImageIndex] +"?"+d.getTime());
}

function selectImageByNumber(index)
{
	if(index < 0 || index > Images.length)
	{
		alert('Invalid value ' + index);
		return;
	}
	var d = new Date();
	ImageIndex = index;
	clearTimeout(ContinueSwapping);
	$("#"+Change_Image).attr('src', Images[ImageIndex] +"?"+d.getTime());
}

function confirmSelectionImage(imageChange)
{
	var d = new Date();
	var idx = document.getElementById(imageChange).value;
	clearTimeout(ContinueSwapping);
	selectImageByNumber(idx);
}

function confirmSelectionImageToChange(imageToChange)
{
	var idx = document.getElementById(imageToChange).value;
	clearTimeout(ContinueSwapping);
	selectImageByNumber(idx);
}

//TODO WITHOUT INDEX
function confirmSelectionImageToAddWithIndex(imageToAdd)
{
	var img = document.getElementById(imageToAdd).value;
	clearTimeout(ContinueSwapping);
	updateImageListWithElement(ListImageTmp[img]);
}

function clearImageList()
{
	var elem = document.getElementById(PlayStop_Btn);
	elem.textContent = PlayText;
	clearTimeout(ContinueSwapping);
	Images = [];
}

*/
