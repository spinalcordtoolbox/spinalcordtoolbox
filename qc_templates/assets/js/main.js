
$(document).ready(function(){
    "use strict;"
    //preload the last generated contrast
    if($("#idToPreload").attr("data-value")){
        changeFrame($("#idToPreload").attr("data-value"));
    }
    // Dropdown automatic hover
    $('.nav-tabs .dropdown').hover(function() {
        $(this).find('.dropdown-menu').first().stop(true, true).slideDown(150);
    }, function() {
        $(this).find('.dropdown-menu').first().stop(true, true).slideUp(105)
    });

    //bind on click event to display an iframe content

    $('.frame-content').click(function(e){
         changeFrame($(this).attr('id'));
    })
    function changeFrame(id){
        var w_height = window.innerHeight;
        var url = id += '.html';
        if($('iframe').attr('src') == null){
            var elementFrame = $('<iframe id="idIframe" width="100%"  ></iframe>');
            $('#frameInsertHere').append(elementFrame)
        }
        $('iframe').attr('height',w_height);
        $('iframe').attr('src', url);
    }

    (function copyrightYear(){
       var d = new Date();
       var y = d.getFullYear();
       var ele = document.getElementById("copyright");
       ele.innerHTML = '&#169 NeuroPoly. All Rights Reserved ' + y;
       ele.style.marginLeft = "10px";
    })();

});





