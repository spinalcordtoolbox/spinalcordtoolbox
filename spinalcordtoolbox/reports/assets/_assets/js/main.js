var selected_row =null;
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

  function newScroll(newRow)
  {
    var rowTop = newRow.position().top;
    var rowBottom = rowTop + newRow.height();
    var $table = $('.fixed-table-body'); // store instead of calling twice
    var tableHeight = $table.height();
    var currentScroll = $table.scrollTop();
    
    if (rowTop < 0)
    {
        // scroll up
        $('.fixed-table-body').scrollTop(currentScroll + rowTop - 20);
    }
    else if (rowBottom  > tableHeight)
    {
        // scroll down
        var scrollAmount = rowBottom - tableHeight;
        $('.fixed-table-body').scrollTop(currentScroll + scrollAmount + 20);
    }
    
    return false;
  }

  function getActiveColumns()
  {
    var cols = $("table th");
    var col_name = [];
    for(let i=0; i<cols.length; i++)
    {
      if(cols[i].style.display == "")
      {
        col_name.push(cols[i].dataset.field);
      }
    }
    return col_name;
  }

  function check_element(obj,cols,vals)
  {

    for( let i=0; i<cols.length; i++)
    {
      if (obj[cols[i]]!=vals[i])
      {
        return false
      }
    }
    return true
  }

  $("#table").on("click", "tr", function() {
    var index = $(this).index();
    var list = $("#table").bootstrapTable('getData');
    var item = list[index];
    if(!$(this)[0].innerHTML.includes("<th")){
    selected_row = $(this)[0].innerText;
    $("#sprite-img").attr("src", item.background_img).removeClass().addClass(item.orientation);
    $("#overlay-img").attr("src", item.overlay_img).removeClass().addClass(item.orientation);
    document.getElementById("cmdLine").innerHTML = "<b>Command:</b> " + item.cmdline;
    document.getElementById("sctVer").innerHTML = "<b>SCT version:</b> " + item.sct_version;
    $(this).addClass('active').siblings().removeClass('active');
    }
    if($('table tr.active').length>0){ $('table tr.active')[0].scrollIntoView();}
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
      if (obj.length == 0 || obj.text() === "DateDatasetSubjectPathFileContrastFunctionFunction+ArgsQC") {
        $('#table tr:first-child').click();
      }
      else {
        obj.next().click();
      }
      evt.preventDefault(); 
      newScroll(obj)
    }
    // Arrow up: previous subject (or k)
    if (evt.which == 38 || evt.which == 75) {
      if (obj.length == 0) {
        $('#table tr:last-child').click();
      }
      else {
        obj.prev().click();
      }
      evt.preventDefault(); 
      newScroll(obj)
    }
    // f key (mark "failing" subjects using check, X, !)
    if (evt.which == 70) {
      var cols = getActiveColumns();
      var vals = obj[0].innerText.split("\t");
      let rel_index = obj[obj.length - 1].getAttribute("data-index");
      let index = sct_data.findIndex(y => check_element(y,cols,vals))
      const heavy_check_mark = '\u2705'
      const heavy_ballot_x = '\u274C'
      const heavy_excl_mark = '\u26A0\uFE0F'
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
      document.getElementById("table").rows[parseInt(rel_index)+1].className="active";
      selected_row = document.getElementById("table").rows[parseInt(rel_index)+1].innerText;
      document.getElementById("table").rows[parseInt(rel_index)+1].scrollIntoView();
    }
  });

  $("#table").bootstrapTable({
    data: sct_data
  });
});

$("#table").on('search.bs.table', function (e, row) {
  hideColumns();
  rows=$("table tbody tr");
  for(let i=0; i<rows.length; i++)
  {
    if(rows[i].innerText == selected_row)
    {
      rows[i].className="active";
      rows[i].scrollIntoView();
    }
  }
  hideColumns();
});

$("#table").on('sort.bs.table', function (e, row) {
  
  $('#table').on('post-body.bs.table', function(e,params){
    console.log("sort finish");
    $('#table').unbind("post-body.bs.table");
    hideColumns();
    rows=$("table tbody tr");
    for(let i=0; i<rows.length; i++)
    {
      if(rows[i].innerText == selected_row)
      {
        rows[i].className="active";
        rows[i].scrollIntoView();
      }
    }
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