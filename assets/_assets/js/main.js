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

  function build_canvas(qc_details) {

  }

  function build_trials_menu(qc_details) {
    var tpl_items = qc_details.trials.map(x=>`<li><a href="#"> ${x.title} </a></li>`).join('');
    document.getElementById("trials-menu-list").innerHTML = tpl_items;
  }

//  $.getJSON('index.json').done(function(data) {
//    qc_details = data;
//    current_qc = qc_details.trials;
//    document.title = qc_details.title;
//    $("#qc-organization")[0].innerText = qc_details.organization;
//    $("#qc-header")[0].innerText = qc_details.header;
//    build_trials_menu(qc_details);
//    build_canvas(qc_details);
//  });

  $("table").on("click", "tr", function() {
    var index = $(this).index();
    var list = $("table").bootstrapTable('getData');
    $("#sprite-img").attr("src", list[index].background_img);
    $("#overlay-img").attr("src", list[index].overlay_img);

    console.log(list[index]);
  });
});
