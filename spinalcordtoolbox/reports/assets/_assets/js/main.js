var selected_row =null;
if (!Element.prototype.scrollIntoViewIfNeeded) {
  Element.prototype.scrollIntoViewIfNeeded = function (centerIfNeeded) {
    centerIfNeeded = arguments.length === 0 ? true : !!centerIfNeeded;

    var parent = this.parentNode,
        parentComputedStyle = window.getComputedStyle(parent, null),
        parentBorderTopWidth = parseInt(parentComputedStyle.getPropertyValue('border-top-width')),
        parentBorderLeftWidth = parseInt(parentComputedStyle.getPropertyValue('border-left-width')),
        overTop = this.offsetTop - parent.offsetTop < parent.scrollTop,
        overBottom = (this.offsetTop - parent.offsetTop + this.clientHeight - parentBorderTopWidth) > (parent.scrollTop + parent.clientHeight),
        overLeft = this.offsetLeft - parent.offsetLeft < parent.scrollLeft,
        overRight = (this.offsetLeft - parent.offsetLeft + this.clientWidth - parentBorderLeftWidth) > (parent.scrollLeft + parent.clientWidth),
        alignWithTop = overTop && !overBottom;

    if ((overTop || overBottom) && centerIfNeeded) {
      parent.scrollTop = this.offsetTop - parent.offsetTop - parent.clientHeight / 2 - parentBorderTopWidth + this.clientHeight / 2;
    }

    if ((overLeft || overRight) && centerIfNeeded) {
      parent.scrollLeft = this.offsetLeft - parent.offsetLeft - parent.clientWidth / 2 - parentBorderLeftWidth + this.clientWidth / 2;
    }

    if ((overTop || overBottom || overLeft || overRight) && !centerIfNeeded) {
      this.scrollIntoView(alignWithTop);
    }
  };
}
$(document).ready(function(){
  "use strict";
  var qc_details;
  var current_qc;
  updateQcStates();
  // Emoji download button states
  const heavy_check_mark = '\u2705';
  const heavy_ballot_x = '\u274C';
  const heavy_excl_mark = '\u26A0\uFE0F';
  const empty_state = '';

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
    $("#background-img").attr("src", item.background_img).removeClass().addClass(item.plane);
    $("#overlay-img").attr("src", item.overlay_img).removeClass().addClass(item.plane);
    document.getElementById("cmdLine").innerHTML = "<b>Command:</b> " + item.cmdline;
    document.getElementById("sctVer").innerHTML = "<b>SCT version:</b> " + item.sct_version;
    $(this).addClass('active').siblings().removeClass('active');
    }
    if($('table tr.active').length>0){ $('table tr.active')[0].scrollIntoViewIfNeeded(false);}
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
      if (obj.length == 0 || obj.text() === "DateDatasetSubjectPathFileContrastFunctionFunction+ArgsRankQC") {
        obj = $('#table tr:first-child');
        obj.click();
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
        obj = $('#table tr:last-child');
        obj.click();
      }
      else {
        obj.prev().click();
      }
      evt.preventDefault(); 
      newScroll(obj)
    }
    // Keys that update table state (f key, number keys)
    if ((evt.which == 70) || (evt.which >= 48 && evt.which <= 57) || (evt.which >= 96 && evt.which <= 105)) {
      // Only attempt to update the cell value if the object is a row (i.e. if it has length)
      if (obj.length > 0) {
        // Fetch the index into `sct_data` corresponding to the selected row
        var cols = getActiveColumns();
        var vals = obj[0].innerText.split("\t");
        let rel_index = obj[obj.length - 1].getAttribute("data-index");
        let index = sct_data.findIndex(y => check_element(y,cols,vals))
        var uniqueId = sct_data[index].moddate + '_' + sct_data[index].fname_in + '_' + sct_data[index].command;

        // Update the cell value in `sct_data` based on the key pressed
        if (evt.which == 70) {
          // f key => cycle through qc states
          sct_data[index].qc = (
            sct_data[index].qc === heavy_check_mark
            ? heavy_ballot_x
            : sct_data[index].qc === heavy_ballot_x
            ? heavy_excl_mark
            : sct_data[index].qc === heavy_excl_mark
            ? empty_state
            : heavy_check_mark
          );
          // Save QC state to local storage
          localStorage.setItem(uniqueId+"_qc", sct_data[index].qc);
          // Update table display with updated sct_data
          set_download_yml_btn_state(heavy_excl_mark);
          set_download_yml_btn_state(heavy_ballot_x);

        }
        if ((evt.which >= 48 && evt.which <= 57) || (evt.which >= 96 && evt.which <= 105)) {
          // Normalize numpad event codes (to avoid writing non-numeric characters to the table)
          let code = evt.which;
          if (evt.which >= 96 && evt.which <= 105) {
            code -= 48;
          }
          // 0 key, store "None"
          if (code == 48) {
            sct_data[index].rank = ""
          }
          // 1-9 keys (number row, keypad) => store the value directly
          else {
            sct_data[index].rank = String.fromCharCode(code);
          }
          // Save Rank state to local storage
          localStorage.setItem(uniqueId+"_rank", sct_data[index].rank);
        }

        // Refresh the table with the updated data
        $("#table").bootstrapTable({data: sct_data});
        $("#table").bootstrapTable("load", sct_data);
        hideColumns();
        // Set focus on the row that was just updated
        document.getElementById("table").rows[0].classList.remove("active");
        document.getElementById("table").rows[parseInt(rel_index)+1].className="active";
        selected_row = document.getElementById("table").rows[parseInt(rel_index)+1].innerText;
        document.getElementById("table").rows[parseInt(rel_index)+1].scrollIntoViewIfNeeded(false);
      }
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
                      rows[i].scrollIntoViewIfNeeded(false);
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

function downloadQcStates() {
  var qcFlags = {};
  // Fetch all QC flags from the QC column of the table
  sct_data.forEach(function(item, index) {
      var uniqueId = item.moddate + '_' + item.fname_in + '_' + item.command;
      qcFlags[uniqueId+"_qc"] = item.qc;
      qcFlags[uniqueId+"_rank"] = item.rank;
  });
  // Create a blob and trigger a download
  var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(qcFlags));
  var downloadAnchorNode = document.createElement('a');
  downloadAnchorNode.setAttribute("href", dataStr);
  downloadAnchorNode.setAttribute("download", "qc_flags.json");
  document.body.appendChild(downloadAnchorNode);
  downloadAnchorNode.click();
  downloadAnchorNode.remove();
};

function loadAndSetQcStates(event) {
  var file = event.target.files[0];
  var reader = new FileReader();
  reader.onload = function(e) {
      var qcFlags = JSON.parse(e.target.result);
      for (var key in qcFlags) {
          if (qcFlags.hasOwnProperty(key)) {
              localStorage.setItem(key, qcFlags[key]);
          }
          updateQcStates();
      }
  };
  reader.readAsText(file);
}

function updateQcStates() {
    // Load and set QC state from local storage
    // TODO: create a function for this code block
    sct_data.forEach((item, index) => {
      var uniqueId = sct_data[index].moddate + '_' + sct_data[index].fname_in + '_' + sct_data[index].command;
      const savedQcState = localStorage.getItem(uniqueId+"_qc");
      if (savedQcState) {
        item.qc = savedQcState;
      }
      const savedRankState = localStorage.getItem(uniqueId+"_rank");
      if (savedRankState) {
        item.rank = savedRankState;
      }
  });
  // Update table display with updated sct_data
  $("#table").bootstrapTable({data: sct_data});
  $("#table").bootstrapTable("load", sct_data);
  hideColumns();
}
