import React from "react";
import { I18nextProvider } from "react-i18next";
import i18n from "@/utils/translations"; // Import translation config
import { SafeAreaProvider } from "react-native-safe-area-context";
import { Slot } from "expo-router"; // Ensures routing works

export default function App() {
  return (
    <I18nextProvider i18n={i18n}>
      <SafeAreaProvider>
        <Slot />
      </SafeAreaProvider>
    </I18nextProvider>
  );
}
