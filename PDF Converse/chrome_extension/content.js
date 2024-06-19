

chrome.runtime.sendMessage({ action: "content_script_loaded" }, function(response) {
    console.log("Response from background script:", response);
});
