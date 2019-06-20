//
//  TextView.swift
//  Misspelling
//
//  Created by Elia Cereda on 17/06/2019.
//  Copyright © 2019 Giorgia Adorni, Elia Cereda e Nassim Habbash. All rights reserved.
//

import Cocoa

class TextView: NSTextView {

    let spellChecker = SpellChecker()
    
    override func viewWillMove(toWindow newWindow: NSWindow?) {
        super.viewWillMove(toWindow: newWindow)
        
        // Setup the SpellChecker
        self.spellChecker.textView = self
    }
}
