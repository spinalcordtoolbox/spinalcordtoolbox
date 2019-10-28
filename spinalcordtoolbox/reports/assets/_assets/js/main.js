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

  $("#table").on("click", "tr", function() {
    var index = $(this).index();
    var list = $("#table").bootstrapTable('getData');
    var item = list[index];
    $("#sprite-img").attr("src", item.background_img).removeClass().addClass(item.orientation);
    $("#overlay-img").attr("src", item.overlay_img).removeClass().addClass(item.orientation);
    document.getElementById("cmdLine").innerHTML = "<b>Command:</b> " + item.cmdline;
    document.getElementById("sctVer").innerHTML = "<b>SCT version:</b> " + item.sct_version;
    $(this).addClass('active').siblings().removeClass('active');
  });

  $('#prev-img').click( function(event) {
    $('table tr.active').prev().click();
    event.preventDefault();
  });

  $('#next-img').click( function(event) {
    $('table tr.active').next().click();
    event.preventDefault();
  });

  $('html').keydown( function(evt) {
    var obj = $('#table tr.active');
    // Arrow down: next subject
    if (evt.which == 40) {
      if (obj.length == 0 || obj.text() === "DateDatasetSubjectPathFileContrastFunctionFunction+Args") {
        $('#table tr:first-child').click();
      }
      else {
        obj.next().click();
      }
    }
    // Arrow up: previous subject
    if (evt.which == 38) {
      if (obj.length == 0) {
        $('#table tr:last-child').click();
      }
      else {
        obj.prev().click();
      }
    }
  });

  $("#table").bootstrapTable({
    data: sct_data
  });
});

function responseHandler(res) {
  var n;
  for(var i = 0; i < res.length; i++) {
    n = new Date(res[i].moddate);
    res[i].moddate = n.toLocaleString();
  }
  return res;
}
