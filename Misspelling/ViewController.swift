//
//  ViewController.swift
//  Misspelling
//
//  Created by Elia Cereda on 15/06/2019.
//  Copyright © 2019 Giorgia Adorni, Elia Cereda e Nassim Habbash. All rights reserved.
//

import Cocoa

class ViewController: NSViewController {

    @IBOutlet var textView: TextView!
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
    }

    override var representedObject: Any? {
        didSet {
        // Update the view, if already loaded.
        }
    }

    @IBAction func showMostLikelySequence(_ sender: Any) {
        self.textView.spellChecker.mostLikelySequence()
    }
}

