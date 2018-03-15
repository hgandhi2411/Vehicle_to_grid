function footer_hider(event) {
    
    var a = event.currentSlide.attributes.getNamedItem('hide-footer');    
    
    if(a) {
        if(a.nodeValue === 'yes') {
            document.querySelector('.footer').style.visibility = 'hidden';
        } 
    }else {
        document.querySelector('.footer').style.visibility = 'visible';
    }        
}
