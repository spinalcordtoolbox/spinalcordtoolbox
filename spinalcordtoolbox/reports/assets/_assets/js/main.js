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
    // Arrow down: next subject (or j)
    if (evt.which == 40 || evt.which == 74) {
      if (obj.length == 0 || obj.text() === "DateDatasetSubjectPathFileContrastFunctionFunction+Args") {
        $('#table tr:first-child').click();
      }
      else {
        obj.next().click();
      }
    }
    // Arrow up: previous subject (or k)
    if (evt.which == 38 || evt.which == 75) {
      if (obj.length == 0) {
        $('#table tr:last-child').click();
      }
      else {
        obj.prev().click();
      }
    }
    // f key (mark "failing" subjects using check, X, !)
    if (evt.which == 70) {
      let index = obj[obj.length - 1].getAttribute("data-index");
      const heavy_check_mark = '\u2714'
      const heavy_ballot_x = '\u2718'
      const heavy_excl_mark = '\u2757'
      sct_data[index].qc = (
          sct_data[index].qc === heavy_check_mark
          ? heavy_ballot_x
          : sct_data[index].qc === heavy_ballot_x
            ? heavy_excl_mark
            : heavy_check_mark
      );
      set_download_yml_btn_state(heavy_excl_mark);
      set_download_yml_btn_state(heavy_ballot_x);
      $("#table").bootstrapTable({data: sct_data});
      $("#table").bootstrapTable("load", sct_data);
      hideColumns();
      document.getElementById("table").rows[0].classList.remove("active");
      document.getElementById("table").rows[parseInt(index)+1].className="active";
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

function set_download_yml_btn_state(marker) {
  let disabled = true;
  sct_data.forEach(item => {
      if (item.qc === marker) {
        disabled = false;
      }
  });
  if (containsNonLatinCodepoints(marker) === true) {
    // This converts e.g. '\u2718' -> '2718' with corresponding id='download_yaml_btn_2718'
    marker = marker.codePointAt(0).toString(16)
  }
  document.getElementById("download_yaml_btn_".concat(marker)).disabled = disabled;
}

function containsNonLatinCodepoints(s) {
    return /[^\u0000-\u00ff]/.test(s);
}