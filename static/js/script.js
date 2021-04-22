var rotation = 0;
$(function() {

    $("#rright").click(function() {
        rotation = (rotation + 90) % 360;
        $(".pic-view").css({'transform': 'rotate('+rotation+'deg)'});
		
        if(rotation != 0){
            $(".pic-view").css({'max-width': '10%'});
        }else{
            $(".pic-view").css({'max-width': '10%'});
        }

        $('#rotation').val(rotation);
    });
	
    $("#rleft").click(function() {
        rotation = (rotation - 90) % 360;
        $(".pic-view").css({'transform': 'rotate('+rotation+'deg)'});
		
        if(rotation != 0){
            $(".pic-view").css({'max-width': '10%'});
        }else{
            $(".pic-view").css({'max-width': '10%'});
        }
        $('#rotation').val(rotation);
    });
});

function filePreview(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#imgPreview + img').remove();
            $('#imgPreview').after('<img src="'+e.target.result+'" class="pic-view" height="10%" width="10%" />');
            max = Math.max($(".pic-view").width(),$(".pic-view").height());
            $('#imgPreview').css({'width':max, 'height':max});
            rotation = 0;
            $('#rotation').val(rotation);
        };
        reader.readAsDataURL(input.files[0]);
        $('.img-preview').show();
    }else{
        $('#imgPreview + img').remove();
        $('.img-preview').hide();
    }
}

$("#file").change(function (){
    // Image preview
    filePreview(this);
});

$( '#form' )
  .submit( function( e ) {
    var data = new FormData(this);
    data.append("CustomField", "This is some extra data, testing");
    imagebox = $('#imagebox')
    imageboxpre = $('#imagebox_pre')
    loadbox = $('#loading')
    imagebox.attr('src' , '')

    $("#result").text('');
    loadbox.attr('src' , 'https://acegif.com/wp-content/uploads/loading-6.gif')
    
    $.ajax( {
//        url: '/api/v1.0/imgrecognize/?rotation='+rotation,
        url: '/api/v1.0/imgrecognize/?exif&resimg&autorotation&resize&rotation='+rotation,
        type: 'POST',
        enctype: 'multipart/form-data',
        data: data,
        processData: false,
        contentType: false,
        success: function (data) {
            $("#result").text(JSON.stringify(data, null, 2));
            jQuery.each(data, function(i, val) {
                if (i == 'data') {
                    for (var m in val) {
                        console.log("SUCCESS : ", i);
                        bytestring = val[m]['img_res']
                        image = bytestring.split('\'')[1]
                        imagebox.attr('src' , 'data:image/jpeg;base64,'+image)
                        loadbox.attr('src' , '')                                            
                    };
                }
            });
            console.log("SUCCESS : ", data);
        },
    } );
    e.preventDefault();
  } );