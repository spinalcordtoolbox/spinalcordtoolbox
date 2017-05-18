$(document).ready(function(){
  "use strict";
  var qc_details;
  var current_qc;

  function copyrightYear(){
    var d = new Date();
    var y = d.getFullYear();
    var ele = document.getElementById("copyright");
    ele.innerHTML = '&#169 NeuroPoly. All Rights Reserved ' + y;
    ele.style.marginLeft = "10px";
  }

  $("table").on("click", "tr", function() {
    var index = $(this).index();
    var list = $("table").bootstrapTable('getData');
    var item = list[index];
    $("#sprite-img").attr("src", item.background_img).removeClass().addClass(item.orientation);
    $("#overlay-img").attr("src", item.overlay_img).removeClass().addClass(item.orientation);
    $(this).addClass('active').siblings().removeClass('active');
    console.log(list[index]);
  });

  $('html').keydown( function(evt) {
    if (evt.which == 39) {
      $('table tr.active').next().click();
    }
    if (evt.which == 37) {
      $('table tr.active').prev().click();
    }
  });
});
